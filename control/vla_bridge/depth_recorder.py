"""One outstanding depth request; network/GPU waits never run on the motion thread.

Episode rows reference immutable request IDs and per-request result manifests.
Results are persisted by the caller when drained, including at episode shutdown.
Busy cadence ticks are explicit skips; there is no unbounded queue of stale frames.
"""
import queue
import threading
import time
from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class DepthRecorder:
    def __init__(self, client, logger):
        self.client, self.logger = client, logger
        self._results = queue.Queue(maxsize=1)
        self._thread = None
        self._pending = None
        self.stats = dict(submitted=0, complete=0, failed=0, skipped_busy=0)

    def submit(self, left, right, step, source_frames):
        self.drain()
        if self._pending is not None:
            self.stats['skipped_busy'] += 1
            return dict(status='skipped_busy', source_step=step)
        request = dict(request_id=f'depth-{step:06d}', source_step=step,
                       source_frames=source_frames, submitted_at_utc=utc_now(),
                       result_path=f'depth_overhead/{step:06d}.json')
        self._pending = request
        # Own a stable snapshot even if a camera backend reuses its array buffer.
        left, right = left.copy(), right.copy()
        def compute():
            start = time.perf_counter()
            try:
                if hasattr(self.client, 'compute_depth_result'):
                    result = self.client.compute_depth_result(left, right, request['request_id'])
                else:
                    result = dict(depth=self.client.compute_depth(left, right))
                if result.get('depth') is None:
                    result.setdefault('error', 'depth request failed')
            except Exception as exc:
                result = dict(depth=None, error=f'{type(exc).__name__}: {exc}')
            result.update(request_ms=(time.perf_counter() - start) * 1000,
                          completed_at_utc=utc_now())
            self._results.put(result)
        self._thread = threading.Thread(target=compute, name='depth-request', daemon=True)
        self.stats['submitted'] += 1
        self._thread.start()
        return dict(request, status='submitted')

    def drain(self):
        if self._pending is None:
            return
        try:
            result = self._results.get_nowait()
        except queue.Empty:
            return
        self._persist(result)

    def _persist(self, result):
        self.logger.write_depth_result(self._pending, result)
        self.stats['failed' if result.get('depth') is None else 'complete'] += 1
        self._pending = None

    def close(self):
        if self._thread is not None:
            self._thread.join(timeout=getattr(self.client, 'timeout_s', 10.0) + 2.0)
        self.drain()
        if self._pending is not None:
            self._persist(dict(depth=None, error='depth worker exceeded shutdown deadline',
                               completed_at_utc=utc_now()))
        return dict(self.stats)

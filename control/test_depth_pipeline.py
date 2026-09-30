"""Regression coverage for persistent serving, asynchronous motion and provenance."""
import base64
import io
import json
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from vla_bridge.fastfs_server import create_app
from vla_bridge.depth_camera import DepthCameraClient
from vla_bridge.depth_recorder import DepthRecorder
from vla_bridge.io_logger import IOLogger
from vla_bridge import action_contract
from test_depth_camera import _make_calibration
from test_run_vla_episode import FakeStereoCameraCountingReads, ScriptedActionSource
from run_vla_episode import run_episode


class InferenceDouble:
    metadata = dict(checkpoint_sha256='test-checkpoint', units='metres')
    def __init__(self):
        self.calls = []
    def __call__(self, left, right, K, baseline):
        self.calls.append((left.copy(), right.copy(), K, baseline))
        return np.full(left.shape[:2], 0.75, dtype=np.float32)


def payload():
    image = np.full((32, 64, 3), [12, 34, 56], dtype=np.uint8)
    encoded = base64.b64encode(cv2.imencode('.png', image)[1]).decode()
    return dict(left_png_b64=encoded, right_png_b64=encoded,
                intrinsics_flat=[100, 0, 32, 0, 100, 16, 0, 0, 1], baseline_m=.05,
                request_id='depth-000001')


def test_resident_service_reuses_one_model_and_echoes_request_identity():
    infer = InferenceDouble()
    http = create_app(infer).test_client()
    assert http.get('/health').json['protocol_version'] == 1
    for _ in range(2):
        response = http.post('/depth', json=payload())
        assert response.status_code == 200
        body = response.json
        assert body['request_id'] == 'depth-000001'
        assert body['model']['checkpoint_sha256'] == 'test-checkpoint'
        depth = np.load(io.BytesIO(base64.b64decode(body['depth_npy_b64'])), allow_pickle=False)
        np.testing.assert_array_equal(depth, np.full((32, 64), .75, dtype=np.float32))
    assert len(infer.calls) == 2
    np.testing.assert_array_equal(infer.calls[0][0][0, 0], [12, 34, 56])


def test_service_rejects_bad_input_and_does_not_queue_overlapping_requests():
    started, release = threading.Event(), threading.Event()
    infer = InferenceDouble()
    class Slow:
        metadata = infer.metadata
        def __call__(self, *args):
            started.set()
            assert release.wait(3)
            return infer(*args)
    app = create_app(Slow())
    with app.test_client() as http:
        assert http.post('/depth', json={}).status_code == 400
    results = []
    thread = threading.Thread(target=lambda: results.append(app.test_client().post('/depth', json=payload()).status_code))
    thread.start()
    try:
        assert started.wait(2)
        assert app.test_client().post('/depth', json=payload()).status_code == 429
        assert app.test_client().get('/health').status_code == 200
    finally:
        release.set()
        thread.join(3)
    assert results == [200]


def test_real_http_client_server_roundtrip_records_aligned_rgb(tmp_path):
    from werkzeug.serving import make_server
    infer = InferenceDouble()
    server = make_server('127.0.0.1', 0, create_app(infer), threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = DepthCameraClient(_make_calibration(), f'http://127.0.0.1:{server.server_port}/depth')
        assert client.check_ready()['status'] == 'ready'
        left = np.full((720, 1280, 3), 55, dtype=np.uint8)
        right = np.full_like(left, 77)
        with IOLogger(tmp_path, {}) as logger:
            recorder = DepthRecorder(client, logger)
            reference = recorder.submit(left, right, 4, {'left': {'path': 'source-left.png'}})
            assert recorder.close()['complete'] == 1
        manifest = json.loads((tmp_path / reference['result_path']).read_text())
        assert manifest['source_step'] == 4
        assert manifest['units'] == 'metres'
        assert manifest['model']['checkpoint_sha256'] == 'test-checkpoint'
        assert manifest['request_ms'] >= manifest['inference_ms']
        assert manifest['calibration_sha256'] == client.calibration_sha256
        depth = np.load(tmp_path / manifest['depth']['path'])
        rgb = cv2.imread(str(tmp_path / manifest['aligned_left']['path']))
        assert depth.shape == rgb.shape[:2] == (544, 992)
        np.testing.assert_array_equal(rgb, infer.calls[0][0])
    finally:
        server.shutdown()
        thread.join(3)


def test_motion_continues_while_depth_is_blocked_and_last_result_is_drained(tmp_path, mock_robot, mock_calibration_file):
    started, release = threading.Event(), threading.Event()
    class SlowDepth:
        def compute_depth(self, left, right):
            started.set()
            assert release.wait(3)
            return np.full((2, 2), .5, dtype=np.float32)
    class Source(ScriptedActionSource):
        def get_action(self, *args):
            if self._step == 1:
                assert started.wait(2)
                assert not release.is_set()  # second action reached with request outstanding
            if self._step == 2:
                release.set()
            return super().get_action(*args)
    stereo = FakeStereoCameraCountingReads(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2, 3), np.uint8))
    try:
        with IOLogger(tmp_path, {}) as logger:
            run_episode(mock_robot, {}, {}, logger, Source(), 'test', 3, 100,
                        action_contract.load_joint_limits_deg(mock_calibration_file),
                        stereo_camera=stereo, depth_client=SlowDepth(), depth_every_n_steps=1)
    finally:
        release.set()
    rows = [json.loads(x) for x in (tmp_path / 'episode.jsonl').read_text().splitlines()]
    assert len(rows) == 3
    assert rows[1]['depth_frames']['overhead']['status'] == 'skipped_busy'
    result = json.loads((tmp_path / rows[0]['depth_frames']['overhead']['result_path']).read_text())
    assert result['status'] == 'complete'
    assert result['source_frames'] == {k: rows[0]['camera_frames'][k] for k in ('overhead_left', 'overhead_right')}


def test_depth_snapshot_owned_and_failures_are_explicit(tmp_path):
    release = threading.Event()
    class Client:
        def compute_depth(self, left, right):
            assert release.wait(2)
            assert left[0, 0, 0] == 4
            raise ValueError('bad response')
    frame = np.full((2, 2, 3), 4, np.uint8)
    with IOLogger(tmp_path, {}) as logger:
        recorder = DepthRecorder(Client(), logger)
        ref = recorder.submit(frame, frame, 7, {})
        frame[:] = 99
        release.set()
        assert recorder.close()['failed'] == 1
    result = json.loads((tmp_path / ref['result_path']).read_text())
    assert result['status'] == 'failed'
    assert 'bad response' in result['error']
    assert result['depth']['path'] is None


def test_logger_refuses_to_mix_two_episodes(tmp_path):
    with IOLogger(tmp_path, {}):
        pass
    with pytest.raises(FileExistsError):
        IOLogger(tmp_path, {})


def test_send_failure_is_not_recorded_as_execution(tmp_path, mock_robot, mock_calibration_file):
    def fail(_):
        raise ConnectionError('bus unavailable')
    mock_robot.send_action = fail
    with IOLogger(tmp_path, {}) as logger:
        run_episode(mock_robot, {}, {}, logger, ScriptedActionSource(), 'test', 1, 100,
                    action_contract.load_joint_limits_deg(mock_calibration_file))
    row = json.loads((tmp_path / 'episode.jsonl').read_text())
    assert row['executed_action'] == {}
    assert row['action_details']['waypoint_sends'][0]['status'] == 'failed'


def test_notebook_embeds_current_server_and_guide_matches():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / 'control/vla_bridge/policy_server.ipynb').read_text())
    cell = next(c for c in notebook['cells'] if c.get('id') == 'resident-fastfs-source')
    assert ''.join(cell['source']) == '%%writefile /content/fastfs_server.py\n' + (root / 'control/vla_bridge/fastfs_server.py').read_text()
    code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
    assert 'threading.Thread(' not in code
    assert '"scripts/run_demo.py"' not in code


def test_bridge_raw_output_survives_validation_and_is_written(tmp_path, mock_robot, mock_calibration_file):
    from test_robot_client import FakeClientReturnsAction
    from vla_bridge.robot_client import BridgeActionSource
    limits = action_contract.load_joint_limits_deg(mock_calibration_file)
    raw = dict.fromkeys(action_contract.JOINT_ORDER, 0.)
    raw['shoulder_pan'] = 500.
    source = BridgeActionSource(FakeClientReturnsAction(raw), 'test-model', limits)
    with IOLogger(tmp_path, {}) as logger:
        run_episode(mock_robot, {}, {}, logger, source, 'test', 1, 100, limits)
    row = json.loads((tmp_path / 'episode.jsonl').read_text())
    assert row['raw_model_output']['shoulder_pan'] == 500.
    assert row['validated_action']['shoulder_pan'] <= limits['shoulder_pan'][1]
    assert row['validator_flags']
    assert row['action_details']['policy_action']['timestep'] == 0


@pytest.mark.parametrize('response', [{'depth_npy_b64': 'not base64'}, {'depth_npy_b64': ''}, {'bad': 'key'}])
def test_malformed_depth_response_is_logged_as_failure(monkeypatch, response):
    from test_depth_camera import _FakeResponse
    from vla_bridge import depth_camera
    monkeypatch.setattr(depth_camera.requests, 'post', lambda *a, **k: _FakeResponse(response))
    client = DepthCameraClient(_make_calibration(), 'http://fake/depth')
    image = np.zeros((720, 1280, 3), np.uint8)
    result = client.compute_depth_result(image, image, 'depth-1')
    assert result['depth'] is None
    assert result['error']


def test_fastfs_checkpoint_loads_once(monkeypatch, tmp_path):
    import sys
    import types
    import torch
    from vla_bridge.fastfs_server import FastFSModel
    monkeypatch.setitem(sys.modules, 'core.utils.utils', types.SimpleNamespace(InputPadder=object))
    monkeypatch.setitem(sys.modules, 'Utils', types.SimpleNamespace(AMP_DTYPE=torch.float16))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    class Model:
        args = types.SimpleNamespace()
        def cuda(self): return self
        def eval(self): return self
    loads = []
    def load(*args, **kwargs):
        loads.append(args)
        return Model()
    monkeypatch.setattr(torch, 'load', load)
    monkeypatch.setattr(FastFSModel, '__call__', lambda self, left, *a: np.ones(left.shape[:2], np.float32))
    checkpoint = tmp_path / 'model.pth'
    checkpoint.write_bytes(b'fixture')
    model = FastFSModel(checkpoint)
    http = create_app(model).test_client()
    assert http.post('/depth', json=payload()).status_code == 200
    assert http.post('/depth', json=payload()).status_code == 200
    assert len(loads) == 1


def test_capture_drains_to_latest_frame_without_backlog(monkeypatch):
    import time
    from vla_bridge.stereo_camera import _FFmpegAVFoundationCapture
    from vla_bridge import stereo_camera
    # Tiny frames keep this pipe/reader lifecycle test independent of real ffmpeg.
    monkeypatch.setattr(stereo_camera, 'STEREO_HEIGHT', 1)
    monkeypatch.setattr(stereo_camera, 'STEREO_WIDTH', 1)
    monkeypatch.setattr(stereo_camera, '_FRAME_BYTES', 3)
    release = threading.Event()
    class Pipe:
        def __init__(self): self.reads = 0
        def read(self, n):
            self.reads += 1
            if self.reads <= 3: return bytes([self.reads] * 3)
            release.wait(2)
            return b''
        def close(self): pass
    class Process:
        stdout = Pipe()
        def poll(self): return None
        def terminate(self): release.set()
        def wait(self, **kwargs): return 0
    monkeypatch.setattr(stereo_camera.subprocess, 'Popen', lambda *a, **k: Process())
    capture = _FFmpegAVFoundationCapture('fake')
    try:
        with capture._condition:
            assert capture._condition.wait_for(lambda: capture._sequence == 3, timeout=1)
        ok, frame = capture.read()
        assert ok and frame[0, 0, 0] == 3
        assert capture.last_frame_metadata['frame_id'] == 3
        with capture._condition:
            capture._latest = (frame, time.monotonic() - 2, {})
        assert capture.read() == (False, None)
    finally:
        capture.release()
    assert not capture._reader.is_alive()

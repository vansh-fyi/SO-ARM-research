"""Resident FastFS HTTP service. Run with the isolated FastFS Python interpreter.

Adapter follows NVlabs/Fast-FoundationStereo scripts/run_demo.py's model API;
no demo subprocess, GUI, point-cloud construction, or intermediate files.
"""
import argparse
import base64
import hashlib
import io
import threading
import time
from pathlib import Path

import cv2
import numpy as np


class FastFSModel:
    def __init__(self, checkpoint, valid_iters=8, max_disp=192):
        import torch
        from core.utils.utils import InputPadder
        from Utils import AMP_DTYPE

        self.torch, self.padder, self.amp_dtype = torch, InputPadder, AMP_DTYPE
        if not torch.cuda.is_available():
            raise RuntimeError("FastFS requires CUDA in its isolated environment")
        checkpoint = Path(checkpoint).resolve()
        digest = hashlib.sha256()
        with checkpoint.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(block)
        # Research checkpoints serialize the complete model. Only load the
        # operator-selected, trusted NVlabs checkpoint, never request data.
        self.model = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.model.args.valid_iters = valid_iters
        self.model.args.max_disp = max_disp
        self.model.cuda().eval()
        self.valid_iters = valid_iters
        self.metadata = dict(checkpoint=checkpoint.name, checkpoint_sha256=digest.hexdigest(),
                             valid_iters=valid_iters, max_disp=max_disp, units='metres')

    def __call__(self, left_bgr, right_bgr, intrinsics, baseline):
        torch = self.torch
        tensors = [torch.from_numpy(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)).to('cuda', dtype=torch.float32)
                   .permute(2, 0, 1).unsqueeze(0) for x in (left_bgr, right_bgr)]
        padding = self.padder(tensors[0].shape, divis_by=32, force_square=False)
        left, right = padding.pad(*tensors)
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=self.amp_dtype):
            disparity = self.model.forward(left, right, iters=self.valid_iters,
                                           test_mode=True, optimize_build_volume='pytorch1')
        disparity = padding.unpad(disparity.float()).cpu().numpy().reshape(left_bgr.shape[:2])
        depth = np.full(disparity.shape, np.nan, dtype=np.float32)
        valid = np.isfinite(disparity) & (disparity > 0)
        np.divide(float(intrinsics[0]) * baseline, disparity, out=depth, where=valid)
        return depth


def create_app(infer):
    from flask import Flask, jsonify, request
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
    inference_lock = threading.Lock()

    @app.get('/health')
    def health():
        return jsonify(status='ready', protocol_version=1, model=infer.metadata)

    @app.post('/depth')
    def depth():
        if not inference_lock.acquire(blocking=False):
            return jsonify(error='depth service busy'), 429
        try:
            body = request.get_json()
            frames = []
            for key in ('left_png_b64', 'right_png_b64'):
                raw = base64.b64decode(body[key], validate=True)
                frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError('invalid PNG')
                frames.append(frame)
            if frames[0].shape != frames[1].shape or max(frames[0].shape[:2]) > 992:
                raise ValueError('expected equal stereo images with dimensions <=992')
            K = np.asarray(body['intrinsics_flat'], dtype=float)
            baseline = float(body['baseline_m'])
            if K.shape != (9,) or not np.isfinite(K).all() or K[0] <= 0 or not np.isfinite(baseline) or baseline <= 0:
                raise ValueError('invalid calibration')
            started = time.perf_counter()
            result = infer(*frames, K.tolist(), baseline)
            elapsed = (time.perf_counter() - started) * 1000
            buf = io.BytesIO()
            np.save(buf, np.asarray(result, dtype=np.float32), allow_pickle=False)
            return jsonify(depth_npy_b64=base64.b64encode(buf.getvalue()).decode('ascii'),
                           model=infer.metadata, inference_ms=elapsed,
                           request_id=body.get('request_id'))
        except (ValueError, KeyError, TypeError) as exc:
            return jsonify(error=str(exc)), 400
        finally:
            inference_lock.release()
    return app


def main():
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--port', type=int, default=3000)
    parser.add_argument('--warmup-width', type=int, default=800)
    parser.add_argument('--warmup-height', type=int, default=576)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.repo).resolve()))
    model = FastFSModel(args.checkpoint)
    # Compilation happens before the socket opens and readiness is reported.
    warmup = np.zeros((args.warmup_height, args.warmup_width, 3), dtype=np.uint8)
    model(warmup, warmup, [800., 0., 400., 0., 800., 288., 0., 0., 1.], .05)
    print('FastFS model loaded and warmed; ready', flush=True)
    create_app(model).run(host='0.0.0.0', port=args.port, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()

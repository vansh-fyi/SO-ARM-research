"""
Local CPU validation harness for the SOARM SO101 robosuite/LIBERO integration
(Phase 2, D-09: all physics/MJCF iteration happens locally, no GPU).

Runs one or more PASS/FAIL checks against the adapted SOARM assets and classes:

    conda run -n libero python explorations/soarm_sanity.py --check compile
    conda run -n libero python explorations/soarm_sanity.py --check all

Check modes:
    compile  - arm + gripper MJCF compile standalone under MuJoCo 2.3.7
    model    - MountedSoarm101 / SoarmGripper registered and instantiable
    reset    - LIBERO env reset with contact forces < 10 N (SC-1)
    render   - agentview + eye_in_hand frames saved, non-black, right-side-up
    soak     - N random-action steps x N seeds without exception or NaN qpos
    tasks    - 3 candidate libero_spatial BDDL tasks step without crashing
    all      - every check above, in order

Exits 0 only if all requested checks PASS. Heavy imports (mujoco, robosuite,
libero) are lazy inside the check functions so --help works everywhere.
"""

import argparse
import os
import sys

import numpy as np

# Point Python at the LIBERO package (repo-root LIBERO dir, NOT
# explorations/LIBERO — create_scene.py's own join points at a
# nonexistent path; the parent dir is the correct target).
LIBERO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "LIBERO"))
sys.path.insert(0, LIBERO_PATH)

# Repo root too: parts of the LIBERO fork's working tree import via the
# absolute "LIBERO.libero..." package prefix, which only resolves when the
# repo root (the LIBERO dir's parent) is on sys.path as a namespace-package
# anchor. python -c snippets get this for free via cwd (''); a script's
# sys.path[0] is explorations/, so insert it explicitly.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

os.environ["MUJOCO_GL"] = "glfw"               # macOS headless via GLFW

import matplotlib
matplotlib.use("Agg")

ARM_XML = os.path.join(
    LIBERO_PATH, "libero/libero/assets/robots/soarm101/robot.xml"
)
GRIPPER_XML = os.path.join(
    LIBERO_PATH, "libero/libero/assets/grippers/soarm_gripper.xml"
)
BDDL_DIR = os.path.join(LIBERO_PATH, "libero/libero/bddl_files/libero_spatial")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# Final libero_spatial tasks (D-03 resolved in plan 02-04): the three whose
# akita_black_bowl_1 init regions sit inside SO101's 0.479 m horizontal reach
# from the tuned base (-0.38, 0) with the most margin:
#   table_center           (-0.075, 0.00) -> 0.305 m (margin +0.174)
#   between_plate_ramekin  (-0.050, 0.20) -> 0.386 m (margin +0.093)
#   ramekin_region         (-0.200, 0.20) -> 0.269 m (margin +0.210)
# The 02-01 candidate next_to_the_plate was swapped out: its bowl region
# (0.01, 0.31) is 0.498 m from the base — beyond the 0.479 m reach.
TASKS = [
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl",
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate.bddl",
]


def _report(ok, name, reason=""):
    """Print the PASS/FAIL line for a check and pass the result through."""
    if ok:
        print(f"PASS — {name}")
    else:
        print(f"FAIL — {name}: {reason}")
    return ok


def _build_env(bddl_path, camera_size=256):
    """Create a SOARM LIBERO env (lazy libero import)."""
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        robots=["Soarm101"],
        camera_heights=camera_size,
        camera_widths=camera_size,
        has_renderer=False,
        has_offscreen_renderer=True,
    )


def _reset_and_settle(env, settle_steps=10):
    """Reset the env and run zero-action settle steps; return final obs."""
    obs = env.reset()
    for _ in range(settle_steps):
        obs, _, _, _ = env.step(np.zeros(7))
    return obs


def max_contact_force(env):
    """Peak contact force magnitude (N) across all active contacts.

    Exact recipe from 02-RESEARCH.md Code Examples: iterate d.ncon,
    decode each contact wrench with mj_contactForce, take the peak
    norm of the translational component.
    """
    import mujoco

    sim = env.sim
    m, d = sim.model._model, sim.data._data
    peak = 0.0
    for i in range(d.ncon):
        f6 = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, f6)
        peak = max(peak, float(np.linalg.norm(f6[:3])))
    return peak


def check_compile(args):
    """Arm and gripper MJCF each compile standalone under MuJoCo 2.3.7."""
    import mujoco

    ok = True
    for label, path in (("arm", ARM_XML), ("gripper", GRIPPER_XML)):
        if not os.path.isfile(path):
            ok = _report(False, f"compile ({label})", f"missing XML: {path}") and ok
            continue
        try:
            mujoco.MjModel.from_xml_path(path)
            _report(True, f"compile ({label})")
        except Exception as exc:  # schema violation, bad mesh path, etc.
            ok = _report(False, f"compile ({label})", f"{path}: {exc}") and ok
    if not ok:
        return _report(False, "compile", "one or more XML stages failed")
    return _report(True, "compile")


def check_model(args):
    """SOARM classes registered in robosuite mappings and instantiable."""
    try:
        import libero.libero.envs.robots  # noqa: F401 — triggers registration
        from robosuite.robots import ROBOT_CLASS_MAPPING
        from robosuite.models.grippers import GRIPPER_MAPPING

        assert "MountedSoarm101" in ROBOT_CLASS_MAPPING, (
            "MountedSoarm101 not in ROBOT_CLASS_MAPPING"
        )
        assert "SoarmGripper" in GRIPPER_MAPPING, (
            "SoarmGripper not in GRIPPER_MAPPING"
        )

        from libero.libero.envs.robots.soarm import MountedSoarm101
        from libero.libero.envs.grippers.soarm_gripper import SoarmGripper

        MountedSoarm101(idn=0)   # proves arm XML path + joint count match
        SoarmGripper(idn=0)      # proves gripper XML path resolves
        return _report(True, "model")
    except Exception as exc:
        return _report(False, "model", str(exc))


def check_reset(args):
    """Env resets with peak contact force < 10 N after settling (SC-1)."""
    try:
        bddl = os.path.join(BDDL_DIR, args.bddl)
        env = _build_env(bddl, camera_size=args.camera_size)
        _reset_and_settle(env)
        peak = max_contact_force(env)
        print(f"  peak contact force after settle: {peak:.3f} N")
        env.close()
        if peak >= 10.0:
            return _report(False, "reset", f"peak contact force {peak:.3f} N >= 10 N")
        return _report(True, "reset")
    except Exception as exc:
        return _report(False, "reset", str(exc))


def check_render(args):
    """Both camera frames save as non-black, right-side-up RGB images."""
    try:
        from PIL import Image

        os.makedirs(OUT_DIR, exist_ok=True)
        bddl = os.path.join(BDDL_DIR, args.bddl)
        env = _build_env(bddl, camera_size=args.camera_size)
        obs = _reset_and_settle(env)

        frames = {
            "soarm_agentview.png": obs["agentview_image"][::-1],
            "soarm_eye_in_hand.png": obs["robot0_eye_in_hand_image"][::-1],
        }
        ok = True
        for name, frame in frames.items():
            if frame.ndim != 3 or frame.shape[2] != 3:
                ok = _report(
                    False, f"render ({name})", f"bad shape {frame.shape}"
                ) and ok
                continue
            if float(frame.mean()) <= 5:
                ok = _report(
                    False, f"render ({name})",
                    f"frame is (near-)black, mean={frame.mean():.2f}",
                ) and ok
                continue
            out = os.path.join(OUT_DIR, name)
            Image.fromarray(frame).save(out)
            print(f"Saved → {out}")
        env.close()
        if not ok:
            return _report(False, "render", "one or more frames failed")
        return _report(True, "render")
    except Exception as exc:
        return _report(False, "render", str(exc))


def check_soak(args):
    """Random-action soak: --steps steps x --seeds seeds, no NaN/exception."""
    try:
        bddl = os.path.join(BDDL_DIR, args.bddl)
        for seed in range(args.seeds):
            np.random.seed(seed)
            env = _build_env(bddl, camera_size=args.camera_size)
            env.reset()
            for step in range(args.steps):
                env.step(np.random.uniform(-1, 1, 7))
                if step % 100 == 0:
                    assert np.isfinite(env.sim.data.qpos).all(), (
                        f"non-finite qpos at seed={seed} step={step}"
                    )
            env.close()
            print(f"  seed {seed}: {args.steps} steps OK")
        return _report(True, "soak")
    except Exception as exc:
        return _report(False, "soak", str(exc))


def check_tasks(args):
    """Each candidate libero_spatial task steps 50 random actions, no crash."""
    ok = True
    for bddl_name in TASKS:
        try:
            env = _build_env(
                os.path.join(BDDL_DIR, bddl_name), camera_size=args.camera_size
            )
            env.reset()
            for _ in range(50):
                env.step(np.random.uniform(-1, 1, 7))
            env.close()
            _report(True, f"tasks ({bddl_name})")
        except Exception as exc:
            ok = _report(False, f"tasks ({bddl_name})", str(exc)) and ok
    if not ok:
        return _report(False, "tasks", "one or more tasks failed")
    return _report(True, "tasks")


CHECKS = {
    "compile": check_compile,
    "model": check_model,
    "reset": check_reset,
    "render": check_render,
    "soak": check_soak,
    "tasks": check_tasks,
}


def main():
    parser = argparse.ArgumentParser(
        description="SOARM SO101 local CPU sanity checks (Phase 2, D-09)."
    )
    parser.add_argument(
        "--check",
        required=True,
        choices=["compile", "model", "reset", "render", "soak", "tasks", "all"],
        help="which check to run (all = every check in order)",
    )
    parser.add_argument(
        "--bddl",
        default="pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl",
        help="libero_spatial BDDL filename used by reset/render/soak checks",
    )
    parser.add_argument("--steps", type=int, default=500,
                        help="random-action steps per seed in the soak check")
    parser.add_argument("--seeds", type=int, default=3,
                        help="number of seeds in the soak check")
    parser.add_argument("--camera-size", type=int, default=256,
                        help="square camera height/width for env rendering")
    args = parser.parse_args()

    names = list(CHECKS) if args.check == "all" else [args.check]
    results = [CHECKS[name](args) for name in names]

    failed = [name for name, ok in zip(names, results) if not ok]
    if failed:
        print(f"\n{len(failed)}/{len(names)} checks FAILED: {', '.join(failed)}")
        sys.exit(1)
    print(f"\n{len(names)}/{len(names)} checks PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()

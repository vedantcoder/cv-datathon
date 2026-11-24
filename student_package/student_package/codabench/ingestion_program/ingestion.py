"""Codabench ingestion program: runs a student's code submission on the hidden test set.

    python ingestion.py <input_dir> <output_dir> <program_dir> <submission_dir>

input_dir      : questions.json + images/<image_id>.jpg   (test data; NO answers)
program_dir    : this folder; also holds a copy of the datathon `eval/` package and config.yaml
submission_dir : the unzipped student submission: model.py (build_model) + checkpoint.pt (+ helper files)
output_dir     : predictions.json, complexity.json, status.json, run_info.json  -> read by the scoring program

Participants cannot see this program's stdout/stderr (hide_output). Failures are therefore reported only
as a numeric status code in status.json (no messages, so test data cannot leak through error text).
Students get the full traceback by running the same program locally: eval/check_submission.py.
"""
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

STATUS = {
    0: "ok",
    1: "submission files missing (need model.py and checkpoint.pt)",
    2: "importing model.py or build_model() failed",
    3: "torch.load / load_state_dict failed",
    4: "complexity measurement failed (predict on the dummy input, or predict did not return list[str])",
    5: "predict() raised an error on test data",
    6: "predict() returned the wrong number or type of answers",
}
CKPT_NAME = "checkpoint.pt"


def find_submission_root(sub: Path) -> Path:
    """Accept model.py at the zip root, or inside a single top-level folder."""
    if (sub / "model.py").exists():
        return sub
    dirs = [d for d in sub.iterdir() if d.is_dir() and not d.name.startswith(("__", "."))]
    if len(dirs) == 1 and (dirs[0] / "model.py").exists():
        return dirs[0]
    return sub


MAX_THREADS = 2


def cpu_quota():
    """CPUs allowed by the cgroup quota (docker --cpus), or None. Affinity alone over-reports in containers."""
    for path, parse in (("/sys/fs/cgroup/cpu.max", lambda t: t.split()),                         # cgroup v2
                        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", lambda t: [t.strip(), None])):     # cgroup v1
        try:
            quota, period = parse(Path(path).read_text())
            if quota in ("max", "-1"):
                return None
            if period is None:
                period = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip()
            return max(1, int(int(quota) / int(period)))
        except (OSError, ValueError):
            continue
    return None


def env_info() -> dict:
    try:
        n_cpu = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cpu = os.cpu_count()
    q = cpu_quota()
    if q:
        n_cpu = min(n_cpu, q)
    ram = None
    try:
        with open("/proc/meminfo") as f:
            ram = round(int(f.readline().split()[1]) / 1e6, 1)
    except OSError:
        pass
    try:  # cgroup memory limit, if any
        lim = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if lim != "max":
            ram = min(ram or 1e9, round(int(lim) / 1e9, 1))
    except (OSError, ValueError):
        pass
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=3).close()
        net = 1
    except OSError:
        net = 0
    return {"n_cpus": n_cpu, "ram_gb": ram, "network": net}


def run(input_dir: Path, output_dir: Path, program_dir: Path, submission_dir: Path, verbose: bool = True) -> int:
    """Returns the status code; writes all outputs. verbose=True prints full tracebacks (local checks)."""
    t_start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(program_dir))
    import torch
    from eval.complexity import count_flops, count_params, dummy_inputs, sha256_file
    from eval.run_predict import predict_split
    import yaml

    info = env_info()
    # Cap at MAX_THREADS: Codabench public workers report 6 CPUs but they are shared. Probe 2026-11-11 (baseline,
    # 10.4k q): 6 threads 2264 s, 1 thread 655 s, 2 threads 395 s. Submissions may still call torch.set_num_threads.
    torch.set_num_threads(int(os.environ.get("VQA_DT_THREADS", 0)) or max(1, min(info["n_cpus"] or 1, MAX_THREADS)))
    info["torch_threads"] = torch.get_num_threads()
    info["torch_version"] = torch.__version__
    cfg = yaml.safe_load(open(program_dir / "config.yaml"))
    status, stage_times = 0, {}

    def fail(code):
        if verbose:
            traceback.print_exc()
            print(f"STATUS {code}: {STATUS[code]}", file=sys.stderr)
        return code

    def done(code):
        info.update(status=code, total_sec=round(time.time() - t_start, 1), **stage_times)
        (output_dir / "status.json").write_text(json.dumps({"status": code}))
        (output_dir / "run_info.json").write_text(json.dumps(info))
        print(f"ingestion finished: status {code} ({STATUS[code]}) in {info['total_sec']}s; {info}", flush=True)
        return code

    sub = find_submission_root(submission_dir)
    ckpt = sub / CKPT_NAME
    if not (sub / "model.py").exists() or not ckpt.exists():
        return done(fail(1) if verbose else 1)
    sys.path.insert(1, str(sub))

    # build + load
    t0 = time.time()
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("student_model", sub / "model.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        torch.manual_seed(0)
        model = mod.build_model()
    except Exception:
        return done(fail(2))
    try:
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model = model.to("cpu").eval()
    except Exception:
        return done(fail(3))
    stage_times["load_sec"] = round(time.time() - t0, 1)

    # complexity (same procedure as eval/complexity.py)
    t0 = time.time()
    try:
        img, q = dummy_inputs(cfg)
        complexity = {"params": int(count_params(model)), "flops": count_flops(model, img, q),
                      "checkpoint_sha256": sha256_file(ckpt), "torch_version": torch.__version__}
    except Exception:
        return done(fail(4))
    (output_dir / "complexity.json").write_text(json.dumps(complexity))
    stage_times["complexity_sec"] = round(time.time() - t0, 1)

    # predictions on the test set
    t0 = time.time()
    questions = json.load(open(input_dir / "questions.json"))
    try:
        # num_workers=0: DataLoader workers pass batches through /dev/shm, which is only 64 MB in a default
        # Docker container (Codabench workers) -> "No space left on device". JPEG decoding is cheap anyway.
        preds = predict_split(model, questions, input_dir / "images", batch_size=32, num_workers=0,
                              device="cpu", log_every=50 if verbose else 0)
    except RuntimeError as e:
        if "predict returned" in str(e):
            return done(fail(6))
        return done(fail(5))
    except Exception:
        return done(fail(5))
    if len(preds) != len(questions) or not all(isinstance(p["answer"], str) for p in preds):
        return done(fail(6))
    (output_dir / "predictions.json").write_text(json.dumps(preds))
    stage_times["predict_sec"] = round(time.time() - t0, 1)
    return done(status)


def main():
    if len(sys.argv) != 5:
        sys.exit("usage: ingestion.py <input_dir> <output_dir> <program_dir> <submission_dir>")
    paths = [Path(p).resolve() for p in sys.argv[1:5]]
    code = run(*paths, verbose=os.environ.get("VQA_DT_VERBOSE") == "1")
    # Always exit 0 so the scoring step runs and the status code reaches the leaderboard.
    sys.exit(0 if code is not None else 1)


if __name__ == "__main__":
    main()

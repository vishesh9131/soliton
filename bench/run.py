"""Memory-predictability benchmark: Soliton vs PyTorch vs JAX vs TensorFlow, GPT-2 124M, fp32, naive attention.

E1 predict-then-run: each framework estimates peak memory before training (if it has a way to), then trains a
   few steps; we record its allocator peak, the device-level peak (NVML), throughput and loss.
E2 budget: the question a user actually has -- "what is the largest batch that trains under X GiB?" Frameworks
   with a predictor answer it by bisecting predictions and then verify with one real run; everyone is also
   bisected with real runs to get the ground truth and the cost of trial-and-error.

    python bench/run.py --gpu 3 --experiments e1,e2

The GPU must be otherwise idle: device-level peaks are measured as NVML used memory above the idle baseline.
"""
import argparse
import importlib.metadata as md
import json
import os
import subprocess
import sys
import threading
import time

import pynvml

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = {"soliton": "soliton_gpt2.py", "pytorch": "torch_gpt2.py", "jax": "jax_gpt2.py", "tensorflow": "tf_gpt2.py"}
CAN_PREDICT = ("soliton", "pytorch")
GiB = 2**30
# TensorFlow runs from its own venv (tensorflow[and-cuda]) because its CUDA wheels clash with PyTorch's.
TF_PYTHON = os.environ.get("TF_PYTHON", os.path.join(HERE, "..", "..", ".venvs", "tf", "bin", "python"))


class GpuPeak:
    """Peak of device-level used memory above the idle baseline, polled every 10 ms."""

    def __init__(self, index):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(index)

    def used(self):
        return pynvml.nvmlDeviceGetMemoryInfo(self.h).used

    def __enter__(self):
        self.base, self.peak, self.stop = self.used(), 0, False
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        return self

    def _poll(self):
        while not self.stop:
            self.peak = max(self.peak, self.used() - self.base)
            time.sleep(0.01)

    def __exit__(self, *exc):
        self.stop = True
        self.thread.join()


PRECISION = "fp32"  # set from --precision; every framework is held to the same one


def run_args(fw, args):
    """Flags that only apply to real training runs, not predictions."""
    if PRECISION == "tf32" and fw in ("soliton", "pytorch", "tensorflow"):
        return args + ["--tf32"]
    return args


def launch(fw, args, gpu, extra_env=None, visible=True, timeout=3600):
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID",
               CUDA_VISIBLE_DEVICES=str(gpu) if visible else "", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    if PRECISION == "fp32":  # JAX's default lets XLA use TF32; the env var alone does not stop it
        env.update(NVIDIA_TF32_OVERRIDE="0", JAX_DEFAULT_MATMUL_PRECISION="highest")
    else:
        env.update(NVIDIA_TF32_OVERRIDE="1")
    env.update(extra_env or {})
    python = TF_PYTHON if fw == "tensorflow" else sys.executable
    t0 = time.time()
    with GpuPeak(gpu) as g:
        try:
            p = subprocess.run([python, os.path.join(HERE, SCRIPTS[fw]), *args], env=env,
                               capture_output=True, text=True, timeout=timeout)
            stdout, stderr = p.stdout, p.stderr
        except subprocess.TimeoutExpired as e:
            stdout, stderr = e.stdout or "", f"timeout after {timeout}s"
    res = [ln for ln in stdout.splitlines() if ln.startswith("RESULT ")]
    if res:
        r = json.loads(res[-1][len("RESULT "):])
    else:  # crashed without reporting; an allocator failure still counts as OOM
        low = stderr.lower()
        oom = any(s in low for s in ("out of memory", "resource_exhausted", "resourceexhausted", "oom"))
        r = dict(framework=fw, oom=oom, crash=not oom, error=stderr.strip().splitlines()[-1][:300] if stderr.strip() else "")
    r.update(wall_s=time.time() - t0, gpu_peak=g.peak)
    return r


def e1(fws, batches, gpu, steps, out):
    rows = []
    for fw in fws:
        for b in batches:
            row = dict(framework=fw, batch=b)
            if fw in CAN_PREDICT:
                # Soliton predicts with no GPU visible at all; PyTorch's FakeTensorMode needs a CUDA device.
                pr = launch(fw, ["--predict", "--batch", str(b)], gpu, visible=fw != "soliton")
                # PyTorch's MemTracker gives one allocation-level number; Soliton gives allocated and reserved.
                row.update(predicted=pr.get("predicted"), predicted_alloc=pr.get("predicted_alloc", pr.get("predicted")),
                           predict_s=pr.get("predict_s"),
                           predict_wall_s=pr["wall_s"], predict_gpu_bytes=pr["gpu_peak"], predict_error=pr.get("error"))
            r = launch(fw, run_args(fw, ["--batch", str(b), "--steps", str(steps)]), gpu)
            row.update({k: r.get(k) for k in ("oom", "crash", "error", "native_peak", "native_peak_alloc", "gpu_peak",
                                              "tok_s", "step_ms", "first_step_s", "loss_first", "loss_last", "wall_s")})
            rows.append(row)
            print("E1 " + json.dumps(row), flush=True)
            save(rows, out, lambda r: (r["framework"], r["batch"]))


def load(path):
    return json.load(open(path)) if os.path.exists(path) else []


def save(rows, path, key):
    """Merge into the results file, replacing rows with the same key, so frameworks can be re-run separately."""
    fresh = {key(r) for r in rows}
    json.dump([r for r in load(path) if key(r) not in fresh] + rows, open(path, "w"), indent=1)


def bisect(ok, lo, hi):
    """Largest b in [lo, hi] with ok(b), assuming ok is monotone decreasing. 0 if none."""
    if not ok(lo):
        return 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid - 1)
    return lo


def e2(fws, gpu, budget_gib, hi, out):
    total = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(gpu)).total
    rows = []
    for fw in fws:
        row = dict(framework=fw, budget_gib=budget_gib)
        runs = []

        def trains(b):
            if fw == "jax":  # JAX's cap is a preallocated fraction of the device
                r = launch(fw, ["--batch", str(b), "--steps", "3"], gpu,
                           {"XLA_PYTHON_CLIENT_PREALLOCATE": "true",
                            "XLA_PYTHON_CLIENT_MEM_FRACTION": f"{budget_gib * GiB / total:.4f}"})
            else:
                r = launch(fw, run_args(fw, ["--batch", str(b), "--steps", "3", "--budget-gib", str(budget_gib)]), gpu)
            runs.append(dict(batch=b, ok=not (r.get("oom") or r.get("crash")), wall_s=r["wall_s"], error=r.get("error")))
            print(f"E2 {fw} batch {b}: {'trains' if runs[-1]['ok'] else 'OOM/crash'} ({r['wall_s']:.0f}s)", flush=True)
            return runs[-1]["ok"]

        if fw in CAN_PREDICT:
            preds = []

            def predicted_fits(b):
                if fw == "soliton":  # dry run under the budget itself: exact yes/no
                    r = launch(fw, ["--predict", "--batch", str(b), "--budget-gib", str(budget_gib)], gpu, visible=False)
                    ok = bool(r.get("fits"))
                else:  # MemTracker gives a peak; compare it with the budget
                    r = launch(fw, ["--predict", "--batch", str(b)], gpu)
                    ok = r.get("predicted") is not None and r["predicted"] <= budget_gib * GiB
                preds.append(dict(batch=b, predicted=r.get("predicted"), fits=ok, wall_s=r["wall_s"]))
                return ok

            t0 = time.time()
            b_pred = bisect(predicted_fits, 1, hi)
            row.update(predicted_max=b_pred, predict_search_s=time.time() - t0, predict_calls=preds)
            row["predicted_max_trains"] = trains(b_pred) if b_pred else None
            row["predicted_max_plus1_ooms"] = not trains(b_pred + 1)
            runs.clear()

        t0 = time.time()
        row["true_max"] = bisect(trains, 1, hi)
        row.update(real_search_s=time.time() - t0, real_search_runs=len(runs),
                   real_search_ooms=sum(not r["ok"] for r in runs), real_runs=runs)
        rows.append(row)
        print("E2 " + json.dumps({k: v for k, v in row.items() if k not in ("real_runs", "predict_calls")}), flush=True)
        save(rows, out, lambda r: r["framework"])


def e3(gpu, budget_gib, hi, out):
    """Soliton only: largest batch under the budget when it may also choose activation checkpointing,
    decided purely by dry runs, then verified by training under the hard cap."""
    preds = []

    def fits(b):
        r = launch("soliton", ["--predict", "--autofit", "--batch", str(b), "--budget-gib", str(budget_gib)], gpu,
                   visible=False)
        preds.append(dict(batch=b, fits=bool(r.get("fits")), checkpoint=r.get("checkpoint"),
                          predicted=r.get("predicted"), wall_s=r["wall_s"]))
        print(f"E3 plan batch {b}: {'fits' if preds[-1]['fits'] else 'does not fit'} (checkpoint {r.get('checkpoint')})", flush=True)
        return preds[-1]["fits"]

    t0 = time.time()
    b = bisect(fits, 1, hi)
    plan_s = time.time() - t0
    chosen = next((p for p in reversed(preds) if p["batch"] == b and p["fits"]), {})
    r = launch("soliton", run_args("soliton", ["--autofit", "--batch", str(b), "--steps", "6",
                                               "--budget-gib", str(budget_gib)]), gpu)
    row = dict(framework="soliton+autofit", budget_gib=budget_gib, max_batch=b, checkpoint=chosen.get("checkpoint"),
               predicted=chosen.get("predicted"), plan_s=plan_s, plan_calls=preds,
               trains=not (r.get("oom") or r.get("crash")), native_peak=r.get("native_peak"), tok_s=r.get("tok_s"),
               error=r.get("error"))
    print("E3 " + json.dumps({k: v for k, v in row.items() if k != "plan_calls"}), flush=True)
    save([row], out, lambda x: x["framework"])


def gib(n):
    return "—" if n is None else f"{n / GiB:.2f}"


def report(e1_rows, e2_rows, e3_rows, meta, path):
    lines = ["# Soliton benchmark: memory predictability", "",
             f"GPU: {meta['gpu']} · driver {meta['driver']} · " + " · ".join(f"{k} {v}" for k, v in meta["versions"].items()),
             "", f"Model: GPT-2 124M, seq 1024, causal attention, AdamW, grad clip 1.0, random tokens. "
             f"Matmul precision: **{meta['precision']}** for every framework "
             f"({'true fp32, tensor cores off' if meta['precision'] == 'fp32' else 'TF32 tensor cores allowed'}).", ""]
    if e1_rows:
        lines += ["## E1 · Predict before running, then run", "",
                  "Errors compare the prediction with the framework's own allocator peaks: allocated (live tensors) and "
                  "reserved (what the allocator holds from the driver, which is what decides OOM).", "",
                  "| framework | batch | predicted GiB | actual alloc / reserved GiB | error vs alloc | error vs reserved "
                  "| predict time | GPU memory used to predict | NVML peak GiB | tok/s |",
                  "|---|---|---|---|---|---|---|---|---|---|"]

        def err(p, a):
            return f"{(p - a) / a * 100:+.2f}%" if p and a else "—"

        for r in e1_rows:
            has_pred = r.get("predict_s") is not None
            actual = "OOM" if r.get("oom") else f"{gib(r.get('native_peak_alloc'))} / {gib(r.get('native_peak'))}"
            cells = [r["framework"], r["batch"], gib(r.get("predicted")) if has_pred else "no API", actual,
                     err(r.get("predicted_alloc"), r.get("native_peak_alloc")), err(r.get("predicted"), r.get("native_peak")),
                     f"{r['predict_s']:.2f}s" if has_pred else "—", gib(r.get("predict_gpu_bytes")) if has_pred else "—",
                     gib(r.get("gpu_peak")), "—" if r.get("oom") else f"{r.get('tok_s') or 0:,.0f}"]
            lines.append("| " + " | ".join(map(str, cells)) + " |")
        lines.append("")
    if e2_rows:
        lines += [f"## E2 · Largest batch under a {e2_rows[0]['budget_gib']} GiB budget", "",
                  "| framework | answer from prediction | time, no training runs | correct? | true max (real bisection) | real bisection time | runs / OOMs |",
                  "|---|---|---|---|---|---|---|"]
        for r in e2_rows:
            if "predicted_max" in r:
                correct = r["predicted_max"] == r["true_max"]
                pred = f"{r['predicted_max']} | {r['predict_search_s']:.0f}s | {'yes' if correct else 'no'}"
            else:
                pred = "no API | — | —"
            lines.append(f"| {r['framework']} | {pred} | {r['true_max']} | {r['real_search_s']:.0f}s | "
                         f"{r['real_search_runs']} / {r['real_search_ooms']} |")
        lines.append("")
    for r in e3_rows:
        lines += [f"## E3 · Soliton auto-fit under the same {r['budget_gib']} GiB budget", "",
                  "Soliton may also pick how many blocks to checkpoint (recompute in backward). The choice is made by "
                  "dry runs only, then verified by training under the hard cap.", "",
                  "| framework | largest batch | blocks checkpointed | planning time, no GPU | trains under cap "
                  "| peak reserved predicted / actual GiB | tok/s |",
                  "|---|---|---|---|---|---|---|",
                  f"| {r['framework']} | {r['max_batch']} | {r['checkpoint']} / 12 | {r['plan_s']:.0f}s | "
                  f"{'yes' if r['trains'] else 'no'} | {gib(r.get('predicted'))} / {gib(r.get('native_peak'))} | "
                  f"{r.get('tok_s') or 0:,.0f} |", ""]
    open(path, "w").write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True, help="physical index (nvidia-smi order)")
    ap.add_argument("--frameworks", default="soliton,pytorch,jax,tensorflow")
    ap.add_argument("--experiments", default="e1,e2")
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--budget-gib", type=float, default=24)
    ap.add_argument("--max-batch", type=int, default=48)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--precision", default="fp32", choices=("fp32", "tf32"),
                    help="fp32 holds every framework to true fp32 matmuls; tf32 lets them all use tensor cores")
    args = ap.parse_args()
    global PRECISION
    PRECISION = args.precision
    os.makedirs(args.out, exist_ok=True)
    fws = args.frameworks.split(",")
    h = (pynvml.nvmlInit(), pynvml.nvmlDeviceGetHandleByIndex(args.gpu))[1]
    meta = dict(gpu=pynvml.nvmlDeviceGetName(h), driver=pynvml.nvmlSystemGetDriverVersion(), precision=args.precision,
                versions={k: md.version(k) for k in ("torch", "jax", "tensorflow")})
    e1_path, e2_path, e3_path = (os.path.join(args.out, f"e{i}.json") for i in (1, 2, 3))
    if "e1" in args.experiments:
        e1(fws, [int(b) for b in args.batches.split(",")], args.gpu, args.steps, e1_path)
    if "e2" in args.experiments:
        e2(fws, args.gpu, args.budget_gib, args.max_batch, e2_path)
    if "e3" in args.experiments:
        e3(args.gpu, args.budget_gib, 64, e3_path)
    order = {f: i for i, f in enumerate(SCRIPTS)}
    e1_rows = sorted(load(e1_path), key=lambda r: (order[r["framework"]], r["batch"]))
    e2_rows = sorted(load(e2_path), key=lambda r: order[r["framework"]])
    report(e1_rows, e2_rows, load(e3_path), meta, os.path.join(args.out, "REPORT.md"))
    print(open(os.path.join(args.out, "REPORT.md")).read())


if __name__ == "__main__":
    main()

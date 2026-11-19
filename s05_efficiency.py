"""
S05 - Computational complexity and deployment measurements  (R1.5, R2.6)

For MRV-Net and each available saved baseline:
  parameters (total / trainable), weight size (MB, float32), file size on disk,
  FLOPs per image (TF profiler on the frozen graph),
  GPU latency, batch 1 (median and 95th percentile over 200 runs after 30 warm-up runs),
  GPU throughput, batch 32 (images/s),
  CPU latency, batch 1 (median over 50 runs),
  peak GPU memory during batch-32 inference.
"""
import os
import time

import numpy as np
import pandas as pd

import config as C
from common import ensure_dirs, load_model, available_baselines, preprocess, tf, gpu_name, save_json


def flops(model, size):
    try:
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        fn = tf.function(lambda x: model(x, training=False))
        cf = fn.get_concrete_function(tf.TensorSpec([1, size, size, 3], tf.float32))
        frozen = convert_variables_to_constants_v2(cf)
        g = frozen.graph
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        opts["output"] = "none"
        info = tf.compat.v1.profiler.profile(graph=g, run_meta=tf.compat.v1.RunMetadata(), cmd="op", options=opts)
        return info.total_float_ops  # multiply-adds counted as 2 FLOPs
    except Exception as e:
        print("FLOP count failed:", e)
        return np.nan


def timeit(fn, x, warm, runs):
    for _ in range(warm):
        fn(x)
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        y = fn(x)
        _ = np.asarray(y)  # block until done
        ts.append(time.perf_counter() - t)
    return np.array(ts)


def measure(name, path, mode, size):
    m = load_model(path)
    fn = tf.function(lambda x: m(x, training=False))
    x1 = tf.constant(preprocess(np.random.uniform(0, 255, (1, size, size, 3)).astype(np.float32), mode))
    x32 = tf.constant(preprocess(np.random.uniform(0, 255, (32, size, size, 3)).astype(np.float32), mode))
    row = dict(model=name, input=f"{size}x{size}x3", total_params=int(m.count_params()),
               trainable_params=int(sum(np.prod(w.shape) for w in m.trainable_weights)),
               weights_MB=round(m.count_params() * 4 / 2 ** 20, 2),
               file_MB=round(os.path.getsize(path) / 2 ** 20, 2),
               GFLOPs=round(flops(m, size) / 1e9, 2))
    has_gpu = bool(tf.config.list_physical_devices("GPU"))
    if has_gpu:
        try:
            tf.config.experimental.reset_memory_stats("GPU:0")
        except Exception:
            pass
        t1 = timeit(fn, x1, 30, 200)
        t32 = timeit(fn, x32, 5, 30)
        row.update(gpu_latency_ms_median=1e3 * np.median(t1), gpu_latency_ms_p95=1e3 * np.percentile(t1, 95),
                   gpu_throughput_img_s=32 / np.median(t32))
        try:
            row["gpu_peak_memory_MB"] = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2 ** 20
        except Exception:
            row["gpu_peak_memory_MB"] = np.nan
    tf.keras.backend.clear_session()
    return row


def cpu_latency_interleaved(models, rounds=3, runs=40):
    """Single-image CPU latency; models are timed in turn over several rounds (median of all runs),
    so that CPU frequency scaling and run order do not favour any model."""
    loaded = {}
    for name, (path, mode, size) in models.items():
        m = load_model(path)
        fn = tf.function(lambda x, m=m: m(x, training=False))
        x1 = tf.constant(preprocess(np.random.uniform(0, 255, (1, size, size, 3)).astype(np.float32), mode))
        with tf.device("/CPU:0"):
            timeit(fn, x1, 10, 1)
        loaded[name] = (fn, x1)
    times = {n: [] for n in loaded}
    for _ in range(rounds):
        for n, (fn, x1) in loaded.items():
            with tf.device("/CPU:0"):
                times[n] += list(timeit(fn, x1, 3, runs))
    return {n: (1e3 * np.median(t), 1e3 * np.percentile(t, 95)) for n, t in times.items()}


def torch_gpu_timing(models):
    """GPU latency/throughput/memory with the Keras PyTorch backend (TensorFlow has no GPU on native Windows).
    Inputs are moved to the GPU once; timing covers the forward pass only (torch.cuda.synchronize)."""
    import torch
    dev = torch.device("cuda")
    rows = []
    for name, (path, mode, size) in models.items():
        m = load_model(path)
        x1 = torch.as_tensor(preprocess(np.random.uniform(0, 255, (1, size, size, 3)).astype(np.float32), mode),
                             device=dev)
        x32 = torch.as_tensor(preprocess(np.random.uniform(0, 255, (32, size, size, 3)).astype(np.float32), mode),
                              device=dev)
        w = next((v.value for v in m.weights), None)
        on_gpu = bool(w is not None and getattr(w, "is_cuda", False))

        def run(x):
            with torch.inference_mode():
                y = m(x, training=False)
            torch.cuda.synchronize()
            return y

        for _ in range(50):
            run(x1)
        t1 = []
        for _ in range(300):
            t = time.perf_counter(); run(x1); t1.append(time.perf_counter() - t)
        torch.cuda.reset_peak_memory_stats()
        for _ in range(10):
            run(x32)
        t32 = []
        for _ in range(50):
            t = time.perf_counter(); run(x32); t32.append(time.perf_counter() - t)
        rows.append(dict(model=name, weights_on_gpu=on_gpu, gpu_latency_ms_median=1e3 * np.median(t1),
                         gpu_latency_ms_p95=1e3 * np.percentile(t1, 95),
                         gpu_batch32_ms_median=1e3 * np.median(t32),
                         gpu_throughput_img_s=32 / np.median(t32),
                         gpu_peak_memory_MB=torch.cuda.max_memory_allocated() / 2 ** 20,
                         gpu_name=torch.cuda.get_device_name(0), gpu_backend="keras-torch"))
        print(rows[-1])
        del m, x1, x32
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(os.path.join(C.RESULTS_DIR, "efficiency_gpu.csv"), index=False)


def model_list():
    models = {"MRV-Net": (C.MRVNET_PATH, C.MRVNET_INPUT_MODE, C.IMG_SIZE)}
    for name, path in available_baselines().items():
        models[name] = (path, C.BASELINE_INPUT_MODE[name], C.BASELINE_IMG_SIZE.get(name, C.IMG_SIZE))
    return models


def main():
    import sys
    import subprocess
    ensure_dirs()
    if "--gpu-torch" in sys.argv:
        torch_gpu_timing(model_list())
        return
    rows = [measure(n, p, md, sz) for n, (p, md, sz) in model_list().items()]
    df = pd.DataFrame(rows)
    cpu = cpu_latency_interleaved(model_list())
    df["cpu_latency_ms_median"] = df.model.map(lambda n: cpu[n][0])
    df["cpu_latency_ms_p95"] = df.model.map(lambda n: cpu[n][1])
    # TensorFlow cannot use the GPU on native Windows: time the GPU with the Keras PyTorch backend instead
    if "gpu_latency_ms_median" not in df.columns:
        try:
            import torch
            if torch.cuda.is_available():
                env = dict(os.environ, KERAS_BACKEND="torch")
                subprocess.run([sys.executable, __file__, "--gpu-torch"], env=env, check=True)
                g = pd.read_csv(os.path.join(C.RESULTS_DIR, "efficiency_gpu.csv"))
                df = df.merge(g, on="model", how="left")
        except ImportError:
            pass
    df.to_csv(os.path.join(C.RESULTS_DIR, "efficiency.csv"), index=False)
    import platform
    save_json(dict(gpu=gpu_name(), tf=tf.__version__, platform=platform.platform(),
                   cpu=platform.processor()), os.path.join(C.RESULTS_DIR, "efficiency_env.json"))
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

from pathlib import Path
import time
import numpy as np
import onnxruntime as ort
import soundfile as sf


PROJECT_DIR = Path(r"D:\AI_ANC_Project")

ONNX_PATH = (
    PROJECT_DIR
    / "models"
    / "onnx"
    / "conv_tasnet_anc.onnx"
)

TEST_AUDIO = (
    PROJECT_DIR
    / "dataset"
    / "split"
    / "test"
    / "noisy"
    / "1272-128104-0001_v1_snr10.wav"
)

SAMPLE_RATE = 16000
CHUNK_SIZE = 32000

WARMUP_RUNS = 10
BENCHMARK_RUNS = 100


print("=" * 60)
print("ONNX INFERENCE BENCHMARK")
print("=" * 60)


# ------------------------------------------------------------
# Load audio
# ------------------------------------------------------------

audio, sr = sf.read(
    TEST_AUDIO,
    dtype="float32"
)

if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

audio = audio[:CHUNK_SIZE]

if len(audio) < CHUNK_SIZE:
    audio = np.pad(
        audio,
        (0, CHUNK_SIZE - len(audio))
    )

input_data = audio.reshape(
    1, -1
).astype(np.float32)


print(f"Sample rate : {sr} Hz")
print(f"Input shape : {input_data.shape}")
print(
    f"Chunk size  : "
    f"{CHUNK_SIZE / SAMPLE_RATE:.2f} seconds"
)


# ------------------------------------------------------------
# ONNX Runtime
# ------------------------------------------------------------

session = ort.InferenceSession(
    str(ONNX_PATH),
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name


print()
print("Provider:")
print(session.get_providers())


# ------------------------------------------------------------
# Warm-up
# ------------------------------------------------------------

print()
print(f"Warm-up runs: {WARMUP_RUNS}")

for _ in range(WARMUP_RUNS):

    session.run(
        None,
        {
            input_name: input_data
        }
    )


# ------------------------------------------------------------
# Benchmark
# ------------------------------------------------------------

print(
    f"Benchmark runs: {BENCHMARK_RUNS}"
)

times = []

for _ in range(BENCHMARK_RUNS):

    start = time.perf_counter()

    session.run(
        None,
        {
            input_name: input_data
        }
    )

    end = time.perf_counter()

    times.append(
        end - start
    )


times = np.array(times)

average_time = np.mean(times)
median_time = np.median(times)
min_time = np.min(times)

audio_duration = (
    CHUNK_SIZE / SAMPLE_RATE
)

real_time_factor = (
    audio_duration / average_time
)

latency_ms = (
    average_time * 1000
)


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print()
print("=" * 60)
print("BENCHMARK RESULTS")
print("=" * 60)

print(
    f"Average inference time : "
    f"{latency_ms:.2f} ms"
)

print(
    f"Median inference time  : "
    f"{median_time * 1000:.2f} ms"
)

print(
    f"Minimum inference time : "
    f"{min_time * 1000:.2f} ms"
)

print(
    f"Audio chunk duration   : "
    f"{audio_duration:.2f} sec"
)

print(
    f"Real-time factor       : "
    f"{real_time_factor:.2f}x"
)

if real_time_factor > 1:

    print()
    print("STATUS: REAL-TIME CAPABLE")

else:

    print()
    print("STATUS: NEEDS OPTIMIZATION")


print("=" * 60)
from pathlib import Path
import numpy as np
import torch
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


print("=" * 60)
print("ONNX MODEL VERIFICATION")
print("=" * 60)

# Load audio
audio, sr = sf.read(
    TEST_AUDIO,
    dtype="float32"
)

if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

# Use first 32000 samples
audio = audio[:32000]

# Add batch dimension
input_data = audio.reshape(1, -1).astype(
    np.float32
)

print(f"Input shape: {input_data.shape}")
print(f"Sample rate: {sr} Hz")


# ------------------------------------------------------------
# ONNX
# ------------------------------------------------------------

session = ort.InferenceSession(
    str(ONNX_PATH),
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

onnx_output = session.run(
    None,
    {
        input_name: input_data
    }
)[0]

print(
    f"ONNX output shape: {onnx_output.shape}"
)


# ------------------------------------------------------------
# Basic validation
# ------------------------------------------------------------

if not np.isfinite(onnx_output).all():

    print("ERROR: ONNX output contains NaN/Inf")

else:

    print("ONNX output contains no NaN/Inf.")

print(
    f"ONNX output max: "
    f"{np.max(np.abs(onnx_output)):.6f}"
)

print()
print("=" * 60)
print("ONNX VERIFICATION COMPLETED")
print("=" * 60)
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Import the EXACT model from training script
from train_conv_tasnet import ConvTasNetLite


PROJECT_DIR = Path(r"D:\AI_ANC_Project")

MODEL_PATH = PROJECT_DIR / "models" / "best_conv_tasnet.pth"

TEST_CLEAN = PROJECT_DIR / "dataset" / "split" / "test" / "clean"
TEST_NOISY = PROJECT_DIR / "dataset" / "split" / "test" / "noisy"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "diagnostic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SAMPLE_RATE = 16000


def snr(clean, estimate):

    clean = clean.astype(np.float64)
    estimate = estimate.astype(np.float64)

    length = min(len(clean), len(estimate))

    clean = clean[:length]
    estimate = estimate[:length]

    error = clean - estimate

    signal_power = np.mean(clean ** 2)
    error_power = np.mean(error ** 2)

    return 10 * np.log10(
        (signal_power + 1e-12) /
        (error_power + 1e-12)
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("MODEL DIAGNOSTIC")
print("=" * 60)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

model = ConvTasNetLite(
    channels=128,
    blocks=8
).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded.")
print()


# ============================================================
# FIRST TEST FILE
# ============================================================

clean_path = sorted(
    TEST_CLEAN.glob("*.wav")
)[0]

noisy_path = (
    TEST_NOISY /
    clean_path.name
)

print("Testing:")
print(clean_path.name)
print()


clean, sr1 = sf.read(
    clean_path,
    dtype="float32"
)

noisy, sr2 = sf.read(
    noisy_path,
    dtype="float32"
)

if clean.ndim > 1:
    clean = np.mean(clean, axis=1)

if noisy.ndim > 1:
    noisy = np.mean(noisy, axis=1)


# Match lengths
length = min(
    len(clean),
    len(noisy)
)

clean = clean[:length]
noisy = noisy[:length]


# ============================================================
# NORMALIZE EXACTLY LIKE TRAINING
# ============================================================

max_value = np.max(
    np.abs(noisy)
)

if max_value > 0:

    clean_norm = clean / max_value
    noisy_norm = noisy / max_value

else:

    clean_norm = clean
    noisy_norm = noisy


# ============================================================
# MODEL
# ============================================================

input_tensor = torch.from_numpy(
    noisy_norm.astype(np.float32)
).unsqueeze(0).to(DEVICE)


with torch.no_grad():

    enhanced_tensor = model(
        input_tensor
    )


enhanced = (
    enhanced_tensor
    .squeeze(0)
    .cpu()
    .numpy()
)


# ============================================================
# MATCH OUTPUT LENGTH
# ============================================================

enhanced = enhanced[:length]

if max_value > 0:
    enhanced = enhanced * max_value


# ============================================================
# STATS
# ============================================================

print("AUDIO LENGTHS")
print("-" * 40)

print("Clean   :", len(clean))
print("Noisy   :", len(noisy))
print("Enhanced:", len(enhanced))

print()

print("AMPLITUDE")
print("-" * 40)

print(
    "Clean max:",
    np.max(np.abs(clean))
)

print(
    "Noisy max:",
    np.max(np.abs(noisy))
)

print(
    "Enhanced max:",
    np.max(np.abs(enhanced))
)

print()

print("SNR")
print("-" * 40)

noisy_snr = snr(
    clean,
    noisy
)

enhanced_snr = snr(
    clean,
    enhanced
)

print(
    f"Noisy SNR    : {noisy_snr:.2f} dB"
)

print(
    f"Enhanced SNR : {enhanced_snr:.2f} dB"
)

print(
    f"Improvement  : {enhanced_snr - noisy_snr:+.2f} dB"
)


# ============================================================
# SAVE AUDIO
# ============================================================

clean_out = OUTPUT_DIR / "clean.wav"
noisy_out = OUTPUT_DIR / "noisy.wav"
enhanced_out = OUTPUT_DIR / "enhanced.wav"

sf.write(
    clean_out,
    clean,
    SAMPLE_RATE
)

sf.write(
    noisy_out,
    noisy,
    SAMPLE_RATE
)

# Prevent clipping only for saving
enhanced_save = np.clip(
    enhanced,
    -1.0,
    1.0
)

sf.write(
    enhanced_out,
    enhanced_save.astype(np.float32),
    SAMPLE_RATE
)


print()
print("=" * 60)
print("DIAGNOSTIC COMPLETED")
print("=" * 60)

print("Clean   :", clean_out)
print("Noisy   :", noisy_out)
print("Enhanced:", enhanced_out)

print("=" * 60)
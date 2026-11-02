import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"D:\AI_ANC_Project")

TEST_CLEAN = PROJECT_DIR / "dataset" / "split" / "test" / "clean"
TEST_NOISY = PROJECT_DIR / "dataset" / "split" / "test" / "noisy"

MODEL_PATH = PROJECT_DIR / "models" / "best_conv_tasnet.pth"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "enhanced"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DEPTHWISE SEPARABLE CONVOLUTION
# ============================================================

class DepthwiseSeparableConv(nn.Module):

    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=1
    ):

        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels
        )

        self.pointwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=1
        )

        self.norm = nn.GroupNorm(
            1,
            channels
        )

        self.activation = nn.PReLU()

    def forward(self, x):

        residual = x

        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.activation(x)

        return x + residual


# ============================================================
# CONV-TASNET LITE
# ============================================================

class ConvTasNetLite(nn.Module):

    def __init__(
        self,
        channels=128,
        blocks=8
    ):

        super().__init__()

        self.encoder = nn.Conv1d(
            1,
            channels,
            kernel_size=16,
            stride=8,
            padding=4
        )

        self.input_norm = nn.GroupNorm(
            1,
            channels
        )

        self.separator = nn.Sequential(

            *[
                DepthwiseSeparableConv(
                    channels,
                    kernel_size=3,
                    dilation=2 ** (i % 8)
                )
                for i in range(blocks)
            ]

        )

        self.mask = nn.Sequential(

            nn.Conv1d(
                channels,
                channels,
                kernel_size=1
            ),

            nn.PReLU(),

            nn.Conv1d(
                channels,
                channels,
                kernel_size=1
            ),

            nn.Sigmoid()

        )

        self.decoder = nn.ConvTranspose1d(
            channels,
            1,
            kernel_size=16,
            stride=8,
            padding=4
        )

    def forward(self, noisy):

        # [B, T]
        x = noisy.unsqueeze(1)

        # Encoder
        encoded = self.encoder(x)

        encoded = self.input_norm(encoded)

        # Separator
        separated = self.separator(encoded)

        # Mask
        mask = self.mask(separated)

        # Apply mask
        enhanced = encoded * mask

        # Decoder
        output = self.decoder(enhanced)

        output = output.squeeze(1)

        # Match input length
        output = output[:, :noisy.shape[-1]]

        return output


# ============================================================
# SNR CALCULATION
# ============================================================

def calculate_snr(clean, estimated):

    clean = clean.astype(np.float64)
    estimated = estimated.astype(np.float64)

    noise = clean - estimated

    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power < 1e-12:
        return 100.0

    return 10 * np.log10(
        (signal_power + 1e-12) /
        (noise_power + 1e-12)
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("AI ADAPTIVE NOISE CANCELLATION")
print("MODEL EVALUATION")
print("=" * 60)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print()
print(f"Model: {MODEL_PATH}")
print()


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
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

print("Model loaded successfully.")
print()


# ============================================================
# FIND TEST FILES
# ============================================================

clean_files = sorted(
    TEST_CLEAN.glob("*.wav")
)

print(
    f"Test clean files: {len(clean_files)}"
)

print(
    f"Output folder:\n{OUTPUT_DIR}"
)

print()


# ============================================================
# EVALUATION
# ============================================================

noisy_snr_values = []
enhanced_snr_values = []

successful = 0


for index, clean_path in enumerate(clean_files, start=1):

    noisy_path = TEST_NOISY / clean_path.name

    if not noisy_path.exists():

        print(
            f"[SKIP] No noisy file: "
            f"{clean_path.name}"
        )

        continue


    try:

        # ----------------------------------------------------
        # LOAD AUDIO
        # ----------------------------------------------------

        clean, clean_sr = sf.read(
            clean_path,
            dtype="float32"
        )

        noisy, noisy_sr = sf.read(
            noisy_path,
            dtype="float32"
        )


        if clean_sr != SAMPLE_RATE:
            raise ValueError(
                f"Clean SR = {clean_sr}"
            )

        if noisy_sr != SAMPLE_RATE:
            raise ValueError(
                f"Noisy SR = {noisy_sr}"
            )


        # Stereo -> mono
        if clean.ndim > 1:
            clean = np.mean(
                clean,
                axis=1
            )

        if noisy.ndim > 1:
            noisy = np.mean(
                noisy,
                axis=1
            )


        # Match lengths
        length = min(
            len(clean),
            len(noisy)
        )

        clean = clean[:length]
        noisy = noisy[:length]


        # ----------------------------------------------------
        # NORMALIZE SAME AS TRAINING
        # ----------------------------------------------------

        max_value = np.max(
            np.abs(noisy)
        )

        if max_value > 0:

            clean_input = clean / max_value
            noisy_input = noisy / max_value

        else:

            clean_input = clean
            noisy_input = noisy


        # ----------------------------------------------------
        # MODEL INFERENCE
        # ----------------------------------------------------

        noisy_tensor = torch.from_numpy(
            noisy_input.astype(np.float32)
        ).unsqueeze(0).to(DEVICE)


        with torch.no_grad():

            enhanced_tensor = model(
                noisy_tensor
            )


        enhanced = (
            enhanced_tensor
            .squeeze(0)
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # RESTORE SCALE
        # ----------------------------------------------------

        if max_value > 0:

            enhanced = enhanced * max_value


        # Prevent clipping
        enhanced = np.clip(
            enhanced,
            -1.0,
            1.0
        )


        # ----------------------------------------------------
        # SAVE ENHANCED AUDIO
        # ----------------------------------------------------

        output_path = (
            OUTPUT_DIR /
            clean_path.name
        )

        sf.write(
            output_path,
            enhanced.astype(np.float32),
            SAMPLE_RATE
        )


        # ----------------------------------------------------
        # CALCULATE SNR
        # ----------------------------------------------------

        noisy_snr = calculate_snr(
            clean,
            noisy
        )

        enhanced_snr = calculate_snr(
            clean,
            enhanced
        )


        noisy_snr_values.append(
            noisy_snr
        )

        enhanced_snr_values.append(
            enhanced_snr
        )


        successful += 1


        # Print every 50 files
        if index == 1 or index % 50 == 0:

            improvement = (
                enhanced_snr - noisy_snr
            )

            print(
                f"[{index}/{len(clean_files)}] "
                f"Noisy SNR: {noisy_snr:.2f} dB | "
                f"Enhanced SNR: {enhanced_snr:.2f} dB | "
                f"Improvement: {improvement:+.2f} dB"
            )


    except Exception as e:

        print(
            f"[ERROR] {clean_path.name}: {e}"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)


if successful > 0:

    avg_noisy_snr = np.mean(
        noisy_snr_values
    )

    avg_enhanced_snr = np.mean(
        enhanced_snr_values
    )

    avg_improvement = (
        avg_enhanced_snr -
        avg_noisy_snr
    )


    print(
        f"Successful test files : {successful}"
    )

    print(
        f"Average Noisy SNR     : "
        f"{avg_noisy_snr:.2f} dB"
    )

    print(
        f"Average Enhanced SNR  : "
        f"{avg_enhanced_snr:.2f} dB"
    )

    print(
        f"SNR Improvement       : "
        f"{avg_improvement:+.2f} dB"
    )


print()
print(
    f"Enhanced audio saved to:"
)

print(
    OUTPUT_DIR
)

print("=" * 60)
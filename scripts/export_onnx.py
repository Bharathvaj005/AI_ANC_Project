from pathlib import Path
import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"D:\AI_ANC_Project")

MODEL_PATH = PROJECT_DIR / "models" / "best_conv_tasnet.pth"
ONNX_DIR = PROJECT_DIR / "models" / "onnx"

ONNX_DIR.mkdir(parents=True, exist_ok=True)

ONNX_PATH = ONNX_DIR / "conv_tasnet_anc.onnx"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL
# ============================================================

class DepthwiseSeparableConv(nn.Module):

    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=1
    ):
        super().__init__()

        padding = (
            (kernel_size - 1)
            * dilation
            // 2
        )

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

        x = noisy.unsqueeze(1)

        encoded = self.encoder(x)

        encoded = self.input_norm(
            encoded
        )

        separated = self.separator(
            encoded
        )

        mask = self.mask(
            separated
        )

        enhanced = encoded * mask

        output = self.decoder(
            enhanced
        )

        output = torch.tanh(
            output
        )

        output = output.squeeze(1)

        output = output[
            :,
            :noisy.shape[-1]
        ]

        return output


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("ONNX EXPORT")
print("=" * 60)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

model = ConvTasNetLite().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# DUMMY INPUT
# ============================================================

# 2 seconds of 16 kHz audio
dummy_input = torch.randn(
    1,
    32000,
    device=DEVICE
)


# ============================================================
# EXPORT
# ============================================================

print()
print("Exporting to ONNX...")

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["noisy_audio"],
    output_names=["enhanced_audio"],
    dynamic_axes={
        "noisy_audio": {
            0: "batch",
            1: "samples"
        },
        "enhanced_audio": {
            0: "batch",
            1: "samples"
        }
    },
    opset_version=18,
    dynamo=False
)

print()
print("=" * 60)
print("ONNX EXPORT COMPLETED")
print("=" * 60)

print(f"ONNX model:")
print(ONNX_PATH)

print("=" * 60)
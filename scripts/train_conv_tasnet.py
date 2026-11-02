import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"D:\AI_ANC_Project")

TRAIN_CLEAN = PROJECT_DIR / "dataset" / "split" / "train" / "clean"
TRAIN_NOISY = PROJECT_DIR / "dataset" / "split" / "train" / "noisy"

VAL_CLEAN = PROJECT_DIR / "dataset" / "split" / "val" / "clean"
VAL_NOISY = PROJECT_DIR / "dataset" / "split" / "val" / "noisy"

MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 16000

CHUNK_SECONDS = 2
CHUNK_SIZE = SAMPLE_RATE * CHUNK_SECONDS

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.001

NUM_WORKERS = 0
SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("AI ADAPTIVE NOISE CANCELLATION")
print("Conv-TasNet Style Speech Enhancement")
print("=" * 60)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"GPU Memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

print("=" * 60)


# ============================================================
# DATASET
# ============================================================

class AudioDataset(Dataset):

    def __init__(
        self,
        clean_dir,
        noisy_dir,
        training=True
    ):

        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir)
        self.training = training

        clean_files = sorted(
            self.clean_dir.glob("*.wav")
        )

        self.pairs = []

        for clean_file in clean_files:

            noisy_file = (
                self.noisy_dir /
                clean_file.name
            )

            if noisy_file.exists():

                self.pairs.append(
                    (clean_file, noisy_file)
                )

        print(
            f"Dataset: {self.clean_dir.name} | "
            f"Valid pairs: {len(self.pairs)}"
        )


    def __len__(self):

        return len(self.pairs)


    def load_audio(self, path):

        audio, sr = sf.read(
            path,
            dtype="float32"
        )

        if sr != SAMPLE_RATE:

            raise ValueError(
                f"Wrong sample rate: {sr} Hz in {path}"
            )

        if audio.ndim > 1:

            audio = np.mean(
                audio,
                axis=1
            )

        return torch.from_numpy(audio)


    # ========================================================
    # IMPORTANT FIX:
    # CLEAN + NOISY USE THE SAME START POSITION
    # ========================================================

    def get_aligned_chunk(
        self,
        clean,
        noisy
    ):

        length = min(
            clean.shape[0],
            noisy.shape[0]
        )

        clean = clean[:length]
        noisy = noisy[:length]

        if length >= CHUNK_SIZE:

            if self.training:

                start = random.randint(
                    0,
                    length - CHUNK_SIZE
                )

            else:

                # Deterministic validation
                start = 0

            clean = clean[
                start:start + CHUNK_SIZE
            ]

            noisy = noisy[
                start:start + CHUNK_SIZE
            ]

        else:

            padding = (
                CHUNK_SIZE - length
            )

            clean = F.pad(
                clean,
                (0, padding)
            )

            noisy = F.pad(
                noisy,
                (0, padding)
            )

        return clean, noisy


    def __getitem__(self, index):

        clean_path, noisy_path = (
            self.pairs[index]
        )

        clean = self.load_audio(
            clean_path
        )

        noisy = self.load_audio(
            noisy_path
        )

        # SAME chunk position
        clean, noisy = self.get_aligned_chunk(
            clean,
            noisy
        )

        # Normalize together
        max_value = torch.max(
            torch.abs(noisy)
        )

        if max_value > 0:

            clean = clean / max_value
            noisy = noisy / max_value

        return noisy, clean


# ============================================================
# SI-SNR LOSS
# ============================================================

def si_snr_loss(
    estimated,
    target,
    eps=1e-8
):

    estimated = (
        estimated -
        estimated.mean(
            dim=-1,
            keepdim=True
        )
    )

    target = (
        target -
        target.mean(
            dim=-1,
            keepdim=True
        )
    )

    target_energy = (
        torch.sum(
            target ** 2,
            dim=-1,
            keepdim=True
        ) + eps
    )

    projection = (
        torch.sum(
            estimated * target,
            dim=-1,
            keepdim=True
        )
        / target_energy
    ) * target

    noise = estimated - projection

    ratio = (
        torch.sum(
            projection ** 2,
            dim=-1
        )
        /
        (
            torch.sum(
                noise ** 2,
                dim=-1
            ) + eps
        )
    )

    si_snr = (
        10 *
        torch.log10(
            ratio + eps
        )
    )

    return -si_snr.mean()

def combined_loss(estimated, target):
    si_loss = si_snr_loss(estimated, target)
    l1_loss = F.l1_loss(estimated, target)

    return si_loss + 10.0 * l1_loss


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

        encoded = self.input_norm(
            encoded
        )

        # Separation network
        separated = self.separator(
            encoded
        )

        # Mask
        mask = self.mask(
            separated
        )

        # Apply mask
        enhanced = encoded * mask

        # Decoder
        output = self.decoder(
            enhanced
        )

        # Keep audio amplitude bounded
        output = torch.tanh(output)

        output = output.squeeze(1)

        # Match original length
        output = output[
            :,
            :noisy.shape[-1]
        ]

        return output
# ============================================================
# DATASETS
# ============================================================

train_dataset = AudioDataset(
    TRAIN_CLEAN,
    TRAIN_NOISY,
    training=True
)

val_dataset = AudioDataset(
    VAL_CLEAN,
    VAL_NOISY,
    training=False
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# ============================================================
# MODEL
# ============================================================

model = ConvTasNetLite(
    channels=128,
    blocks=8
).to(DEVICE)

print(
    f"Trainable parameters: "
    f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# MIXED PRECISION
# ============================================================

use_amp = DEVICE.type == "cuda"

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


# ============================================================
# CHECKPOINT
# ============================================================

best_val_loss = float("inf")

best_model_path = (
    MODEL_DIR /
    "best_conv_tasnet.pth"
)


# ============================================================
# TRAINING
# ============================================================

for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    train_loss = 0.0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{EPOCHS}"
    )

    for noisy, clean in progress:

        noisy = noisy.to(
            DEVICE,
            non_blocking=True
        )

        clean = clean.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp
        ):

            enhanced = model(
                noisy
            )

            loss = combined_loss(
                  enhanced,
                  clean
)

        scaler.scale(
            loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        train_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    train_loss /= len(
        train_loader
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for noisy, clean in val_loader:

            noisy = noisy.to(
                DEVICE,
                non_blocking=True
            )

            clean = clean.to(
                DEVICE,
                non_blocking=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp
            ):

                enhanced = model(
                    noisy
                )

                loss = combined_loss(
                    enhanced,
                    clean
                )

            val_loss += loss.item()


    val_loss /= len(
        val_loader
    )


    # ========================================================
    # RESULTS
    # ========================================================

    print()

    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    print(
        f"Train Combined Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Validation Combined Loss: "
        f"{val_loss:.4f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "val_loss": val_loss,
            },
            best_model_path
        )

        print(
            "🔥 Best model saved:"
        )

        print(
            best_model_path
        )


    print(
        "-" * 60
    )


# ============================================================
# FINISHED
# ============================================================

print()

print("=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Model saved at:\n"
    f"{best_model_path}"
)

if torch.cuda.is_available():

    print()

    print(
        f"GPU memory allocated: "
        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )

    print(
        f"GPU memory reserved: "
        f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
    )

print("=" * 60)
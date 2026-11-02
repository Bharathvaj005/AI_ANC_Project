from pathlib import Path
import numpy as np
import soundfile as sf
from pystoi import stoi
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"D:\AI_ANC_Project")

CLEAN_DIR = PROJECT_DIR / "dataset" / "split" / "test" / "clean"
NOISY_DIR = PROJECT_DIR / "dataset" / "split" / "test" / "noisy"
ENHANCED_DIR = PROJECT_DIR / "outputs" / "enhanced"

SAMPLE_RATE = 16000


# ============================================================
# AUDIO LOADER
# ============================================================

def load_audio(path):
    audio, sr = sf.read(path, dtype="float32")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != SAMPLE_RATE:
        raise ValueError(
            f"Wrong sample rate {sr} in {path}"
        )

    return audio


# ============================================================
# MAIN
# ============================================================

clean_files = sorted(CLEAN_DIR.glob("*.wav"))

print("=" * 60)
print("STOI SPEECH INTELLIGIBILITY EVALUATION")
print("=" * 60)

print(f"Test files found: {len(clean_files)}")
print(f"Sample rate: {SAMPLE_RATE} Hz")
print("=" * 60)


noisy_scores = []
enhanced_scores = []

successful = 0
failed = 0


for clean_path in tqdm(clean_files, desc="Calculating STOI"):

    filename = clean_path.name

    noisy_path = NOISY_DIR / filename
    enhanced_path = ENHANCED_DIR / filename

    try:

        if not noisy_path.exists():
            failed += 1
            continue

        if not enhanced_path.exists():
            failed += 1
            continue

        clean = load_audio(clean_path)
        noisy = load_audio(noisy_path)
        enhanced = load_audio(enhanced_path)

        # ----------------------------------------------------
        # Make all signals the same length
        # ----------------------------------------------------

        length = min(
            len(clean),
            len(noisy),
            len(enhanced)
        )

        clean = clean[:length]
        noisy = noisy[:length]
        enhanced = enhanced[:length]

        # ----------------------------------------------------
        # STOI
        # ----------------------------------------------------

        noisy_stoi = stoi(
            clean,
            noisy,
            SAMPLE_RATE,
            extended=False
        )

        enhanced_stoi = stoi(
            clean,
            enhanced,
            SAMPLE_RATE,
            extended=False
        )

        noisy_scores.append(noisy_stoi)
        enhanced_scores.append(enhanced_stoi)

        successful += 1

    except Exception as e:

        failed += 1

        print(
            f"\nError: {filename}"
        )
        print(e)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("STOI EVALUATION COMPLETED")
print("=" * 60)

print(
    f"Successful files : {successful}"
)

print(
    f"Failed files     : {failed}"
)

if successful > 0:

    avg_noisy = np.mean(noisy_scores)
    avg_enhanced = np.mean(enhanced_scores)

    improvement = avg_enhanced - avg_noisy

    print(
        f"Average Noisy STOI     : {avg_noisy:.4f}"
    )

    print(
        f"Average Enhanced STOI  : {avg_enhanced:.4f}"
    )

    print(
        f"STOI Improvement       : {improvement:+.4f}"
    )

else:

    print("No files were successfully evaluated.")


print("=" * 60)
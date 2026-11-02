from pathlib import Path
import random
import shutil

ROOT = Path(r"D:\AI_ANC_Project\dataset")

CLEAN = ROOT / "clean"
NOISY = ROOT / "noisy"

OUTPUT = ROOT / "split"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

random.seed(42)

clean_files = sorted(CLEAN.glob("*.wav"))

print(f"Clean files found: {len(clean_files)}")

pairs = []

for clean_file in clean_files:
    noisy_file = NOISY / clean_file.name

    if noisy_file.exists():
        pairs.append((clean_file, noisy_file))

print(f"Valid clean/noisy pairs: {len(pairs)}")

random.shuffle(pairs)

total = len(pairs)

train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

splits = {
    "train": pairs[:train_end],
    "val": pairs[train_end:val_end],
    "test": pairs[val_end:]
}

for split_name, split_pairs in splits.items():

    clean_out = OUTPUT / split_name / "clean"
    noisy_out = OUTPUT / split_name / "noisy"

    clean_out.mkdir(parents=True, exist_ok=True)
    noisy_out.mkdir(parents=True, exist_ok=True)

    for clean_file, noisy_file in split_pairs:
        shutil.copy2(clean_file, clean_out / clean_file.name)
        shutil.copy2(noisy_file, noisy_out / noisy_file.name)

    print(f"{split_name}: {len(split_pairs)} pairs")

print("\n==============================")
print("DATASET SPLIT COMPLETED")
print("==============================")
print(f"Train: {len(splits['train'])}")
print(f"Val:   {len(splits['val'])}")
print(f"Test:  {len(splits['test'])}")
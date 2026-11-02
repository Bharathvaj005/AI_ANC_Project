import random
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000

SPEECH_FOLDER = PROJECT_DIR / "clean_speech"
NOISE_FOLDER = PROJECT_DIR / "noise"

CLEAN_OUTPUT = PROJECT_DIR / "dataset" / "clean"
NOISY_OUTPUT = PROJECT_DIR / "dataset" / "noisy"

# SNR levels used to create different noise conditions
SNR_LEVELS = [-5, 0, 5, 10, 15]

# Number of noisy samples created for each clean speech file
VERSIONS_PER_SPEECH = 3


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

CLEAN_OUTPUT.mkdir(parents=True, exist_ok=True)
NOISY_OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND AUDIO FILES
# ============================================================

speech_files = list(SPEECH_FOLDER.rglob("*.wav"))
noise_files = list(NOISE_FOLDER.rglob("*.wav"))


# ============================================================
# DISPLAY INFORMATION
# ============================================================

print()
print("======================================")
print("       AI ANC DATASET GENERATION")
print("======================================")

print("Project folder :", PROJECT_DIR)
print("Speech folder  :", SPEECH_FOLDER)
print("Noise folder   :", NOISE_FOLDER)

print()
print("Speech files found :", len(speech_files))
print("Noise files found  :", len(noise_files))
print()


# ============================================================
# CHECK DATASET
# ============================================================

if len(speech_files) == 0:

    print("ERROR: No WAV speech files found!")
    print("Check this folder:")
    print(SPEECH_FOLDER)
    raise SystemExit


if len(noise_files) == 0:

    print("ERROR: No WAV noise files found!")
    print("Check this folder:")
    print(NOISE_FOLDER)
    raise SystemExit


# ============================================================
# FUNCTION: MIX SPEECH + NOISE
# ============================================================

def mix_audio(speech, noise, snr_db):

    # Calculate speech power
    speech_power = np.mean(speech ** 2)

    # Calculate noise power
    noise_power = np.mean(noise ** 2)

    # Avoid division by zero
    if noise_power <= 1e-12:

        return speech.copy()


    # Convert SNR from dB to linear
    snr_linear = 10 ** (snr_db / 10)


    # Calculate required noise scaling
    noise_scale = np.sqrt(
        speech_power /
        (noise_power * snr_linear)
    )


    # Scale noise
    noise = noise * noise_scale


    # Mix speech and noise
    noisy_audio = speech + noise


    # Prevent clipping
    max_value = np.max(
        np.abs(noisy_audio)
    )

    if max_value > 0.99:

        noisy_audio = (
            noisy_audio / max_value
        ) * 0.99


    return noisy_audio


# ============================================================
# DATASET GENERATION
# ============================================================

total_pairs = 0


for index, speech_path in enumerate(speech_files):

    print(
        f"[{index + 1}/{len(speech_files)}] "
        f"{speech_path.name}"
    )


    # --------------------------------------------------------
    # LOAD CLEAN SPEECH
    # --------------------------------------------------------

    try:

        speech, _ = librosa.load(
            speech_path,
            sr=SAMPLE_RATE,
            mono=True
        )

    except Exception as error:

        print(
            "ERROR loading speech:",
            speech_path.name
        )

        print(error)

        continue


    # Skip empty audio
    if len(speech) == 0:

        print("Skipping empty speech file.")

        continue


    # --------------------------------------------------------
    # CREATE MULTIPLE NOISY VERSIONS
    # --------------------------------------------------------

    for version in range(
        VERSIONS_PER_SPEECH
    ):


        # ----------------------------------------------------
        # RANDOMLY SELECT NOISE
        # ----------------------------------------------------

        noise_path = random.choice(
            noise_files
        )


        # ----------------------------------------------------
        # LOAD NOISE
        # ----------------------------------------------------

        try:

            noise, _ = librosa.load(
                noise_path,
                sr=SAMPLE_RATE,
                mono=True
            )

        except Exception as error:

            print(
                "ERROR loading noise:",
                noise_path.name
            )

            print(error)

            continue


        # Skip empty noise
        if len(noise) == 0:

            continue


        # ----------------------------------------------------
        # MAKE NOISE LONG ENOUGH
        # ----------------------------------------------------

        if len(noise) < len(speech):

            repeat_count = int(
                np.ceil(
                    len(speech) /
                    len(noise)
                )
            )

            noise = np.tile(
                noise,
                repeat_count
            )


        # ----------------------------------------------------
        # SELECT RANDOM PART OF NOISE
        # ----------------------------------------------------

        if len(noise) > len(speech):

            max_start = (
                len(noise) -
                len(speech)
            )

            start = random.randint(
                0,
                max_start
            )

            noise = noise[
                start:
                start + len(speech)
            ]

        else:

            noise = noise[
                :len(speech)
            ]


        # ----------------------------------------------------
        # SELECT RANDOM SNR
        # ----------------------------------------------------

        snr_db = random.choice(
            SNR_LEVELS
        )


        # ----------------------------------------------------
        # CREATE NOISY SPEECH
        # ----------------------------------------------------

        noisy_speech = mix_audio(
            speech,
            noise,
            snr_db
        )


        # ----------------------------------------------------
        # CREATE OUTPUT FILE NAME
        # ----------------------------------------------------

        base_name = speech_path.stem

        output_name = (
            f"{base_name}"
            f"_v{version + 1}"
            f"_snr{snr_db}.wav"
        )


        # ----------------------------------------------------
        # SAVE CLEAN SPEECH
        # ----------------------------------------------------

        clean_output_path = (
            CLEAN_OUTPUT /
            output_name
        )

        sf.write(
            clean_output_path,
            speech,
            SAMPLE_RATE
        )


        # ----------------------------------------------------
        # SAVE NOISY SPEECH
        # ----------------------------------------------------

        noisy_output_path = (
            NOISY_OUTPUT /
            output_name
        )

        sf.write(
            noisy_output_path,
            noisy_speech,
            SAMPLE_RATE
        )


        total_pairs += 1


# ============================================================
# FINISHED
# ============================================================

print()
print("======================================")
print("    DATASET GENERATION COMPLETED")
print("======================================")

print()
print(
    "Total training pairs created:",
    total_pairs
)

print()
print("Clean dataset:")
print(CLEAN_OUTPUT)

print()
print("Noisy dataset:")
print(NOISY_OUTPUT)

print()
print("======================================")
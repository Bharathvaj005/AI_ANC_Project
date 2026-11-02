import os
import librosa
import soundfile as sf

# =========================
# SETTINGS
# =========================

SAMPLE_RATE = 16000

CLEAN_INPUT = "../clean_speech"
NOISE_INPUT = "../noise"

CLEAN_OUTPUT = "../clean_speech_16k"
NOISE_OUTPUT = "../noise_16k"

os.makedirs(CLEAN_OUTPUT, exist_ok=True)
os.makedirs(NOISE_OUTPUT, exist_ok=True)


def convert_folder(input_folder, output_folder):
    files = os.listdir(input_folder)

    count = 0

    for filename in files:

        input_path = os.path.join(input_folder, filename)

        if not os.path.isfile(input_path):
            continue

        try:

            print("Converting:", filename)

            audio, sr = librosa.load(
                input_path,
                sr=SAMPLE_RATE,
                mono=True
            )

            output_name = os.path.splitext(filename)[0] + ".wav"

            output_path = os.path.join(
                output_folder,
                output_name
            )

            sf.write(
                output_path,
                audio,
                SAMPLE_RATE
            )

            count += 1

        except Exception as e:

            print("ERROR:", filename)
            print(e)

    print(f"\nConverted {count} files.")


# =========================
# CONVERT SPEECH
# =========================

print("\n==============================")
print("CONVERTING CLEAN SPEECH")
print("==============================")

convert_folder(
    CLEAN_INPUT,
    CLEAN_OUTPUT
)


# =========================
# CONVERT NOISE
# =========================

print("\n==============================")
print("CONVERTING NOISE")
print("==============================")

convert_folder(
    NOISE_INPUT,
    NOISE_OUTPUT
)

print("\nALL AUDIO CONVERSION COMPLETED!")
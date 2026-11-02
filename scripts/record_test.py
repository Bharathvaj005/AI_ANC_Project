import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
DURATION = 5
OUTPUT = "outputs/mic_test.wav"

print("🎤 Recording for 5 seconds...")
print("Speak something now!")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    device=1
)

sd.wait()

sf.write(OUTPUT, audio, SAMPLE_RATE)

print("\n✅ Recording completed!")
print(f"Saved to: {OUTPUT}")
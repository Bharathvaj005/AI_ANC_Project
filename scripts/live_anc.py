import numpy as np
import sounddevice as sd
import onnxruntime as ort
import time

# ==============================
# CONFIGURATION
# ==============================

MODEL_PATH = r"D:\AI_ANC_Project\models\onnx\conv_tasnet_anc.onnx"

SAMPLE_RATE = 16000

# Start with 1 second chunks
CHUNK_SIZE = 16000

INPUT_DEVICE = 1
OUTPUT_DEVICE = 3

# ==============================
# LOAD ONNX MODEL
# ==============================

print("=" * 60)
print("AI ANC LIVE MICROPHONE DEMO")
print("=" * 60)

print("\nLoading ONNX model...")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("Model loaded successfully!")

print(f"Input : {input_name}")
print(f"Output: {output_name}")

# ==============================
# AUDIO CALLBACK
# ==============================

processing_times = []

def callback(indata, outdata, frames, time_info, status):

    if status:
        print("Audio status:", status)

    start_time = time.perf_counter()

    # Microphone input
    audio = indata[:, 0].astype(np.float32)

    # ONNX expects shape: (1, samples)
    model_input = audio.reshape(1, -1)

    # AI enhancement
    enhanced = session.run(
        [output_name],
        {input_name: model_input}
    )[0]

    enhanced = enhanced.reshape(-1)

    # Safety clipping
    enhanced = np.clip(enhanced, -1.0, 1.0)

    # Speaker output
    outdata[:, 0] = enhanced

    # Latency measurement
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    processing_times.append(elapsed_ms)


# ==============================
# START STREAM
# ==============================

print("\nStarting microphone...")
print("Speak into the microphone.")
print("Press Ctrl+C to stop.\n")

try:

    with sd.Stream(
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        dtype="float32",
        channels=1,
        device=(INPUT_DEVICE, OUTPUT_DEVICE),
        callback=callback
    ):

        while True:

            time.sleep(1)

            if processing_times:

                avg = np.mean(processing_times[-10:])

                print(
                    f"\rAI processing time: {avg:.2f} ms",
                    end=""
                )

except KeyboardInterrupt:

    print("\n\nStopping...")

    if processing_times:

        print(
            f"Average AI processing time: "
            f"{np.mean(processing_times):.2f} ms"
        )

print("\nDemo finished.")
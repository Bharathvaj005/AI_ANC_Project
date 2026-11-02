# AI ANC --- Architecture

This document describes the implemented processing path in the
repository scripts.

## 1. System workflow

``` mermaid
flowchart TD
    CS[Clean speech WAV files] --> PREP[Load as mono 16 kHz]
    NZ[Noise WAV files] --> PREP
    PREP --> MIX[Mix noise at selected SNR levels]
    MIX --> PAIRS[Aligned clean / noisy pairs]
    PAIRS --> SPLIT[Train / validation / test split]
    SPLIT --> TRAIN[Conv-TasNet-style training]
    TRAIN --> CKPT[best_conv_tasnet.pth]
    CKPT --> EVAL[Test-set inference]
    EVAL --> WAV[Enhanced WAV files]
    EVAL --> METRICS[Noisy SNR / enhanced SNR / improvement]
    CKPT --> EXPORT[ONNX export]
    EXPORT --> ONNX[conv_tasnet_anc.onnx]
    ONNX --> LIVE[ONNX Runtime live inference]
    MIC[Microphone] --> LIVE
    LIVE --> SPK[Speaker / headphones]
```

## 2. Conv-TasNet-style model

``` mermaid
flowchart LR
    IN[Noisy waveform<br/>B × T] --> UNSQ[Add channel dimension]
    UNSQ --> ENC[1-D convolution encoder<br/>kernel 16, stride 8]
    ENC --> NORM[GroupNorm]
    NORM --> SEP[Temporal separator<br/>8 depthwise-separable blocks]
    SEP --> MASK[1×1 conv + PReLU<br/>1×1 conv + sigmoid]
    MASK --> APPLY[Apply estimated mask]
    NORM --> APPLY
    APPLY --> DEC[1-D transposed convolution decoder]
    DEC --> OUT[Enhanced waveform<br/>B × T]
```

### Block descriptions

-   **Encoder:** converts the mono waveform into a 128-channel learned
    representation.
-   **Normalization:** applies GroupNorm to encoded features.
-   **Temporal separator:** stacks eight depthwise-separable convolution
    blocks. Each block uses a depthwise convolution, pointwise
    convolution, GroupNorm, PReLU, and residual connection. Dilation
    grows exponentially across the block sequence.
-   **Mask estimator:** uses pointwise convolutions, PReLU, and sigmoid
    to estimate a feature mask.
-   **Mask application:** multiplies the encoded representation by the
    estimated mask.
-   **Decoder:** reconstructs a mono enhanced waveform using a
    transposed 1-D convolution.

The ONNX export implementation applies a `tanh` to the decoder output
before returning the waveform. Refer to the script source when comparing
export and training/evaluation paths.

## 3. Training and loss

The training script uses paired noisy and clean audio chunks:

-   Sample rate: 16,000 Hz
-   Chunk length: 2 seconds (32,000 samples)
-   Batch size: 8
-   Epoch limit: 20
-   Learning rate: 0.001
-   Random seed: 42
-   Loss: SI-SNR-based objective combined with the additional loss
    defined in the training script

The clean and noisy chunks are sliced from the same starting position to
preserve alignment. They are normalized together.

## 4. Evaluation

For each test pair, the evaluation script:

1.  Loads clean and noisy WAV files.
2.  Checks that both are sampled at 16 kHz.
3.  Converts stereo signals to mono if needed.
4.  Matches signal lengths and applies the same input scaling.
5.  Runs model inference.
6.  Restores the scale and clips the enhanced waveform.
7.  Saves the enhanced file and calculates noisy-input and
    enhanced-output SNR against the clean reference.

Recorded evaluation result:

  Metric                           Value
  ----------------------- --------------
  Successful test files              567
  Average noisy SNR              5.37 dB
  Average enhanced SNR          11.10 dB
  Average improvement       **+5.72 dB**

## 5. ONNX live inference

``` mermaid
sequenceDiagram
    participant Mic as Microphone
    participant Stream as sounddevice stream
    participant ORT as ONNX Runtime
    participant Speaker as Speaker / headphones
    Mic->>Stream: 16 kHz mono audio chunk
    Stream->>ORT: Float32 input tensor (1, samples)
    ORT->>ORT: Run Conv-TasNet ONNX model
    ORT-->>Stream: Enhanced audio chunk
    Stream->>Speaker: Output audio
```

The current live script uses the ONNX Runtime CPU execution provider,
one-second chunks (16,000 samples), and configurable input/output device
indices. Device indices must be checked on the target computer.

## 6. Data layout

``` text
dataset/split/
├── train/
│   ├── clean/
│   └── noisy/
├── val/
│   ├── clean/
│   └── noisy/
└── test/
    ├── clean/
    └── noisy/
```

A clean WAV file must have a noisy counterpart with the same filename in
the matching split. The full audio datasets are not committed to this
repository.

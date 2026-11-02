# AI Adaptive Noise Cancellation (AI ANC)

A deep-learning speech-enhancement project that uses a lightweight
**Conv-TasNet-style 1-D convolutional network** to reduce background
noise in speech. This repository includes training, evaluation, ONNX
export, benchmarking, and live microphone inference scripts, plus
trained PyTorch and ONNX model files.

> **Scope:** This prototype performs neural speech enhancement on
> microphone/audio signals. It is not a complete physical anti-noise
> headphone controller with a reference microphone and speaker-path
> acoustic feedback loop.

## Highlights

-   Conv-TasNet-style encoder, temporal separator, mask estimator, and
    decoder.
-   Depthwise-separable temporal convolution blocks with dilations.
-   SI-SNR-based training objective (see
    `scripts/train_conv_tasnet.py`).
-   Paired clean/noisy audio preparation and train/validation/test
    evaluation.
-   PyTorch checkpoint and ONNX model export.
-   ONNX Runtime inference for a live microphone-to-speaker demo.
-   SNR-based evaluation and ONNX inference benchmarking.

## Architecture

``` mermaid
flowchart TD
    A[Clean speech recordings] --> C[Pair and prepare audio]
    B[Noise recordings] --> C
    C --> D[Mix speech and noise at selected SNR levels]
    D --> E[Aligned clean/noisy WAV pairs]
    E --> F[Train / validation / test split]
    F --> G[Conv-TasNet-style model training]
    G --> H[Best PyTorch checkpoint]
    H --> I[Evaluation on noisy test audio]
    I --> J[Enhanced WAV output and SNR metrics]
    H --> K[ONNX export]
    K --> L[ONNX Runtime]
    L --> M[Live microphone input]
    M --> L
    L --> N[Enhanced speaker output]
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for model blocks and inference
paths.

## Model design

The model implementation in the training/evaluation/export scripts uses:

1.  **Encoder:** 1-D convolution maps mono waveform samples into learned
    features.
2.  **Input normalization:** Group normalization stabilizes the encoded
    representation.
3.  **Temporal separator:** a stack of depthwise-separable convolution
    blocks with dilations.
4.  **Mask estimator:** pointwise convolutions and sigmoid produce a
    feature mask.
5.  **Decoder:** transposed 1-D convolution reconstructs the enhanced
    waveform.

The training script currently configures 16 kHz mono audio, 2-second
training chunks, batch size 8, up to 20 epochs, and learning rate 0.001.
Review the script configuration before retraining.

## Repository contents

``` text
AI_ANC_Project/
├── models/
│   ├── best_conv_tasnet.pth
│   └── onnx/
│       └── conv_tasnet_anc.onnx
├── scripts/
│   ├── benchmark_onnx.py
│   ├── convert_audio.py
│   ├── create_noisy_audio.py
│   ├── create_spectrogram.py
│   ├── diagnose_model.py
│   ├── evaluate_model.py
│   ├── evaluate_stoi.py
│   ├── export_onnx.py
│   ├── live_anc.py
│   ├── record_test.py
│   ├── split_dataset.py
│   ├── test_microphone.py
│   ├── train_conv_tasnet.py
│   └── verify_onnx.py
├── .gitignore
├── README.md
└── ARCHITECTURE.md
```

Audio datasets and generated output files are intentionally excluded
from this Git repository because the local project data is several
gigabytes. Model files are included.

## Performance results

The recorded evaluation run for this project reported:

  Metric                                      Result
  ----------------------------------- --------------
  Successfully evaluated test files              567
  Average noisy-input SNR                    5.37 dB
  Average enhanced-output SNR               11.10 dB
  Average SNR improvement               **+5.72 dB**

These values describe the recorded test run; they are not a guarantee
for every recording or environment. Results depend on test audio, noise
type, signal level, and evaluation setup.

The evaluation script calculates SNR against the paired clean reference
and writes enhanced WAV files to `outputs/enhanced/`.

## Installation

### 1. Clone the repository

Several scripts currently contain the Windows project path
`D:\AI_ANC_Project`. The simplest way to run them without editing those
paths is to clone to that location:

``` powershell
cd D:\
git clone https://github.com/Bharathvaj005/AI_ANC_Project.git
cd D:\AI_ANC_Project
```

If you clone elsewhere, update the `PROJECT_DIR` / `MODEL_PATH`
constants in the relevant scripts.

### 2. Create and activate a virtual environment

``` powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If Python 3.14 is not installed, use a Python version supported by the
PyTorch package you choose and adjust the `py -3.14` command.

### 3. Install PyTorch with CUDA support

The development environment used a CUDA 12.8 PyTorch build:

``` powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Install the other libraries used by the scripts:

``` powershell
python -m pip install numpy soundfile librosa tqdm onnx onnxruntime sounddevice
```

For CPU-only machines, install the CPU PyTorch build instead. Check the
official [PyTorch installation
selector](https://pytorch.org/get-started/locally/) for a package
matching your Python, operating system, driver, and CUDA requirements.

### 4. Verify PyTorch and CUDA

``` powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The development setup used CUDA-enabled PyTorch `2.11.0+cu128` and an
NVIDIA GeForce RTX 3050 Laptop GPU. Your driver and GPU must support the
PyTorch CUDA package you install.

## Dataset documentation

The full local dataset is **not included** in this repository. Keep
source audio in local storage or distribute it through an appropriate
dataset host, then arrange the files in the expected directories.

### Source audio

-   **Clean speech:** WAV recordings placed under `clean_speech/`.
-   **Noise:** WAV environmental/noise recordings placed under `noise/`.
-   Only use recordings you are permitted to use and redistribute.
    Preserve dataset licenses, attribution, and source information.

### Generate paired clean/noisy samples

`scripts/create_noisy_audio.py` reads WAV files recursively from
`clean_speech/` and `noise/`, loads them as mono 16 kHz audio, and
creates noisy versions at randomly selected SNR levels of -5, 0, 5, 10,
and 15 dB. The current configuration creates three noisy versions per
clean speech file and writes pairs to:

``` text
dataset/
├── clean/
└── noisy/
```

### Train/validation/test layout

The training and evaluation scripts expect aligned clean/noisy WAV pairs
in this structure:

``` text
dataset/
└── split/
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

Each clean file must have a noisy counterpart with the **same filename**
in the corresponding `noisy/` folder. Audio must be mono or convertible
to mono and sampled at **16,000 Hz**.

The repository contains `scripts/split_dataset.py`. Review its paths and
settings before running it. Back up an existing split before
regenerating it.

### Dataset access and licensing

Add your dataset download instructions or dataset-host link here when
you publish the data. Do not redistribute third-party recordings unless
their licenses allow it. Include attribution and license details for
each source dataset.

## Running the project

Run commands from the project root after preparing the dataset folders.

### Train

``` powershell
python scripts/train_conv_tasnet.py
```

The current training script saves the best checkpoint to
`models/best_conv_tasnet.pth`.

### Evaluate and save enhanced audio

``` powershell
python scripts/evaluate_model.py
```

The script reads the test split, reports average noisy/enhanced SNR, and
writes enhanced WAV files to `outputs/enhanced/`.

### Export the trained checkpoint to ONNX

``` powershell
python scripts/export_onnx.py
```

The exported model is written to `models/onnx/conv_tasnet_anc.onnx`.

### Verify / benchmark ONNX

``` powershell
python scripts/verify_onnx.py
python scripts/benchmark_onnx.py
```

Check the test-audio path configured in `scripts/benchmark_onnx.py`
before running the benchmark.

### Live microphone demo

``` powershell
python scripts/live_anc.py
```

The current live demo uses ONNX Runtime's CPU execution provider, 16 kHz
mono audio, and device indices configured in `scripts/live_anc.py`.
Check `INPUT_DEVICE` and `OUTPUT_DEVICE` for your computer. Use
headphones and start at a low output volume to reduce the risk of
feedback.

## Demo: before and after

The evaluation script saves enhanced samples under `outputs/enhanced/`,
but generated audio is excluded from this repository. To publish a
reproducible demo:

1.  Select a short noisy input WAV and its matching enhanced output WAV.
2.  Confirm you have permission to share both recordings.
3.  Copy them into `docs/demo/`, for example `noisy_example.wav` and
    `enhanced_example.wav`.
4.  Add links or an embedded audio player here after committing those
    files.

  Sample               Audio
  -------------------- ------------------------------------------------
  Before enhancement   Add a link to `docs/demo/noisy_example.wav`
  After enhancement    Add a link to `docs/demo/enhanced_example.wav`

No demo audio is currently committed, so these are intentional
placeholders.

## Main scripts

  -----------------------------------------------------------------------
  Script                              Purpose
  ----------------------------------- -----------------------------------
  `train_conv_tasnet.py`              Train the Conv-TasNet-style model

  `evaluate_model.py`                 Evaluate test pairs, calculate SNR,
                                      save enhanced audio

  `create_noisy_audio.py`             Mix clean speech and noise at
                                      selected SNRs

  `split_dataset.py`                  Create train/validation/test splits

  `export_onnx.py`                    Export the PyTorch checkpoint to
                                      ONNX

  `verify_onnx.py`                    Verify the exported ONNX model

  `benchmark_onnx.py`                 Benchmark ONNX inference

  `live_anc.py`                       Live microphone-to-speaker
                                      enhancement using ONNX Runtime

  `evaluate_stoi.py`                  STOI evaluation utility

  `convert_audio.py`                  Audio conversion utility

  `test_microphone.py`,               Microphone and recording tests
  `record_test.py`                    

  `diagnose_model.py`                 Model diagnostics
  -----------------------------------------------------------------------

## Limitations and reproducibility notes

-   This is a research/prototype speech-enhancement system; it may
    introduce artifacts or distort speech for noise conditions unlike
    the training data.
-   The live script is configured for particular Windows audio device
    indices. Device indices vary by machine.
-   Several scripts currently contain absolute Windows paths. Update
    them when using another directory or operating system.
-   Full datasets and generated outputs are not included;
    training/evaluation requires the dataset layout described above.
-   The reported SNR improvement is from one recorded evaluation run.
    Re-run evaluation on your own test set to reproduce or compare
    results.

## License

No project license has been added yet. Add a license only after deciding
how others may use, modify, and redistribute this work. Third-party
datasets may have separate licenses.

## Acknowledgements

Built as an AI-based adaptive noise cancellation / speech-enhancement
project using PyTorch, Conv-TasNet-style temporal convolution, and ONNX
Runtime.

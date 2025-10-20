# Vogent Turn

**Fast and accurate turn detection for voice AI**

Multimodal turn detection that combines audio intonation and text context to accurately determine when a speaker has finished their turn in a conversation.

## Key Features

- **Multimodal**: Uses both audio (Whisper encoder) and text (SmolLM) for context-aware predictions
- **Fast**: Optimized with `torch.compile` for low-latency inference
- **Easy to Use**: Simple Python API with just a few lines of code
- **Production-Ready**: Batched inference, model caching, and comprehensive error handling

## Architecture

- **Audio Encoder**: Whisper-Tiny (processes up to 8 seconds of 16kHz audio)
- **Text Model**: SmolLM-135M (12 layers, ~80M parameters)  
- **Classifier**: Binary classification (turn complete / turn incomplete)

The model projects audio embeddings into the LLM's input space and processes them together with conversation context for turn detection.

---

## Installation

### From PyPI (coming soon)

```bash
pip install vogent-turn
```

### From Source

```bash
git clone https://github.com/vogent/vogent-turn.git
cd vogent-turn
pip install -e .
```

### Requirements

- Python >=3.8
- PyTorch >=2.1.0
- Transformers >=4.35.0
- See `requirements.txt` for full list

---

## Quick Start

### Python Library

```python
from vogent_turn.inference import TurnDetector
import numpy as np

# Initialize detector (compiles model on first run - takes ~30s)
detector = TurnDetector()

# Your audio: mono, float32 in range [-1, 1]. If not 16 kHz, the predict method will resample to 16 kHz.
import soundfile as sf
audio, sr = sf.read("speech.wav")

# Detect turn endpoint with conversation context
result = detector.predict(
    audio,
    prev_line="What is your favorite color?",
    curr_line="I think it's blue",
    sample_rate=sr
)

print(result)
# {'is_endpoint': True, 'prob_endpoint': 0.92, 'prob_continue': 0.08}
```

### CLI Tool

```bash
# Basic usage (sample rate automatically detected from file)
vogent-turn-predict speech.wav \
  --prev "What is your favorite color?" \
  --curr "I think it's blue"

# Use CPU instead of GPU
vogent-turn-predict speech.wav \
  --prev "Hello" \
  --curr "Hi there" \
  --device cpu
```

**Note:** Sample rate is automatically detected from the audio file. Audio will be resampled to 16kHz internally if needed.

---

## API Reference

### `TurnDetector`

Main class for turn detection inference.

#### Constructor

```python
detector = TurnDetector(
    model_name="vogent/Vogent-Turn-80M",  # HuggingFace model ID
    revision="main",                     # Model revision
    device=None,                         # "cuda", "cpu", or None (auto)
    compile_model=True                   # Use torch.compile for speed
)
```

#### `predict()`

Detect if the current speaker has finished their turn.

```python
result = detector.predict(
    audio,                    # np.ndarray: (n_samples,) mono float32
    prev_line="",             # str: Previous speaker's text (optional)
    curr_line="",             # str: Current speaker's text (optional)
    sample_rate=None,         # int: Sample rate in Hz (recommended to specify)
    return_probs=False        # bool: Return probabilities
)
```

**Note:** The model operates at 16kHz internally. If you provide audio at a different sample rate, it will be automatically resampled (requires `librosa`). If no sample rate is specified, 16kHz is assumed with a warning.

**Returns:**
- If `return_probs=False`: `bool` (True = turn complete, False = continue)
- If `return_probs=True`: `dict` with keys:
  - `is_endpoint`: bool
  - `prob_endpoint`: float (0-1)
  - `prob_continue`: float (0-1)

#### `predict_batch()`

Process multiple audio samples efficiently in a single batch.

```python
results = detector.predict_batch(
    audio_batch,              # list[np.ndarray]: List of audio arrays
    context_batch=None,       # list[dict]: List of context dicts with 'prev_line' and 'curr_line'
    sample_rate=None,         # int: Sample rate in Hz (applies to all audio)
    return_probs=False        # bool: Return probabilities
)
```

**Note:** All audio samples in the batch must have the same sample rate. Audio will be automatically resampled to 16kHz if a different rate is specified.

**Returns:**
- List of predictions (same format as `predict()` depending on `return_probs`)

#### Audio Requirements

- **Sample rate**: 16kHz
- **Channels**: Mono
- **Format**: float32 numpy array
- **Range**: [-1.0, 1.0]
- **Duration**: Up to 8 seconds (longer audio will be truncated)

**Example audio loading:**

```python
import soundfile as sf
import librosa

# Load and resample
audio, sr = sf.read("speech.wav")
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

# Convert to mono if stereo
if audio.ndim > 1:
    audio = audio.mean(axis=1)

# Normalize to [-1, 1] if needed
audio = audio.astype(np.float32)
if np.abs(audio).max() > 1.0:
    audio = audio / np.abs(audio).max()
```

### Text Context Format

The model uses conversation context to improve predictions:

- **`prev_line`**: What the previous speaker said (e.g., a question)
- **`curr_line`**: What the current speaker is saying (e.g., their response)

For best performance, do not include terminal punctuation (periods, etc.).

**Example:**
```python
result = detector.predict(
    audio,
    prev_line="How are you doing today?",
    curr_line="I'm doing great thanks"
)
```

---

## Model Details

### Multimodal Architecture

```
Audio (16kHz) ─────> Whisper Encoder ─> Audio Embeddings (1500D)
                                              |
                                              v
                                        Audio Projector
                                              |
                                              v
Text Context ─────> SmolLM Tokenizer ─> Text Embeddings (variable length)
                                              |
                                              v
                              [Audio Embeds + Text Embeds] ─> SmolLM
                                              |
                                              v
                                      Classification Head
                                              |
                                              v
                                    [Endpoint / Continue]
```

### Training Data

The model is trained on conversational audio with labeled turn boundaries. It learns to detect:
- **Prosodic cues**: Pitch, intonation, pauses
- **Semantic cues**: Completeness of thought, question-answer patterns
- **Contextual cues**: Conversation flow and expectations

---

## Examples

### Basic Endpoint Detection

```python
from vogent_turn.inference import TurnDetector
import soundfile as sf

detector = TurnDetector()
audio, sr = sf.read("speech.wav")

# Simple binary decision
is_endpoint = detector.predict(audio, prev_line="What is your phone number?", curr_line="The number is 2148241616", 
    sample_rate=sr)
print(f"Turn complete: {is_endpoint}")
```

### With Confidence Scores

```python
result = detector.predict(audio, prev_line="What is your phone number?", curr_line="The number is 2148241616", sample_rate=sr, return_probs=True)
print(f"Endpoint probability: {result['prob_endpoint']:.2%}")
print(f"Continue probability: {result['prob_continue']:.2%}")

# Use threshold for decision
if result['prob_endpoint'] > 0.8:
    print("High confidence turn endpoint")
```

### Batch Processing

For efficient batch processing of multiple audio samples, use `predict_batch()`:

```python
import soundfile as sf

detector = TurnDetector()

# Load multiple audio files
audio_files = ["recording1.wav", "recording2.wav", "recording3.wav"]
audio_batch = []
sample_rates = []

for file in audio_files:
    audio, sr = sf.read(file)
    audio_batch.append(audio)
    sample_rates.append(sr)

# Prepare context for each audio sample
context_batch = [
    {"prev_line": "How are you?", "curr_line": "I'm doing great"},
    {"prev_line": "What's your name?", "curr_line": "My name is"},
    {"prev_line": "Where are you from?", "curr_line": "I'm from"},
]

# Process all samples in a single batch (more efficient than looping)
# Note: All audio must have the same sample rate for batch processing
results = detector.predict_batch(
    audio_batch,
    context_batch=context_batch,
    sample_rate=sample_rates[0],  # Assumes all audio has same sample rate
    return_probs=True
)

# Print results
for i, result in enumerate(results):
    print(f"{audio_files[i]}: {result['is_endpoint']} "
          f"(confidence: {result['prob_endpoint']:.2%})")
```
---
## Troubleshooting

### Audio format issues

**Solution:** Ensure audio meets requirements:
```python
import soundfile as sf
import librosa
import numpy as np

# Load and convert
audio, sr = sf.read("file.wav")

# Resample to 16kHz
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

# Convert to mono
if audio.ndim > 1:
    audio = audio.mean(axis=1)

# Ensure float32 and normalized
audio = audio.astype(np.float32)
if np.abs(audio).max() > 1.0:
    audio = audio / np.abs(audio).max()
```

---
## Development

### Project Structure

```
vogent_turn/
├── __init__.py              # Package exports
├── inference.py             # Main TurnDetector class
├── predict.py               # CLI tool
├── smollm_whisper.py        # Model architecture
├── whisper.py               # Whisper components
├── requirements.txt         # Dependencies
└── setup.py                 # Package configuration
```

### Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## Citation

If you use this library in your research, please cite:

```bibtex
@software{vogent_turn,
  title = {Vogent Turn: Multimodal Turn Detection for Conversational AI},
  author = {Vogent},
  year = {2024},
  url = {https://github.com/vogent/vogent-turn}
}
```

---

## License

Inference code is open-source under Apache 2.0. Model weights are under a modified Apache 2.0 license with stricter attribution requirements for certain types of usage.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/vogent/vogent-turn/issues)
- **Discussions**: [GitHub Discussions](https://github.com/vogent/vogent-turn/discussions)

---

## Changelog

### v0.1.0 (2025-10-19)
- Initial release
- Multimodal turn detection with Whisper + SmolLM
- Python library and CLI tool
- Torch.compile optimization for fast inference

# Vogent Turn

**Fast and accurate turn detection for voice AI**

Multimodal turn detection that combines audio intonation and text context to accurately determine when a speaker has finished their turn in a conversation.

Technical report can be found [here](https://blog.vogent.ai/posts/voturn-80m-state-of-the-art-turn-detection-for-voice-agents).

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
from vogent_turn import TurnDetector
import soundfile as sf
import urllib.request

# Initialize detector
detector = TurnDetector(compile_model=True, warmup=True)

# Download and load audio
audio_url = "https://storage.googleapis.com/voturn-sample-recordings/incomplete_number_sample.wav"
urllib.request.urlretrieve(audio_url, "sample.wav")
audio, sr = sf.read("sample.wav")

# Run turn detection with conversational context
result = detector.predict(
    audio,
    prev_line="What is your phone number",
    curr_line="My number is 804",
    sample_rate=sr,
    return_probs=True,
)

print(f"Turn complete: {result['is_endpoint']}")
print(f"Confidence: {result['prob_endpoint']:.1%}")
```

### CLI Tool

```bash
# Basic usage (sample rate automatically detected from file)
vogent-turn-predict speech.wav \
  --prev "What is your phone number" \
  --curr "My number is 804"
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
    sample_rate=None,         # int: Sample rate in Hz (recommended to specify, otherwise 16kHz is assumed)
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

### Text Context Format

The model uses conversation context to improve predictions:

- **`prev_line`**: What the previous speaker said (e.g., a question)
- **`curr_line`**: What the current speaker is saying (e.g., their response)

For best performance, do not include terminal punctuation (periods, etc.).

**Example:**
```python
result = detector.predict(
    audio,
    prev_line="How are you doing today",
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

Sample scripts can be found in the `examples/` directory. 
`python3.10 examples/basic_usage.py` downloads an audio file and runs the turn detector. 
`python3.10 examples/batch_processing.py` downloads two audio files and runs the turn detector with a batched input.
`request_batcher.py` is a sample implementation of a thread for continuous receiving and batching of requests (e.g. in a production setting).

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

---

## Changelog

### v0.1.0 (2025-10-19)
- Initial release
- Multimodal turn detection with Whisper + SmolLM
- Python library and CLI tool
- Torch.compile optimization for fast inference

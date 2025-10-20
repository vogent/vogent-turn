"""
Standalone inference wrapper for turn detection model.

Usage:
    from inference import TurnDetector
    
    detector = TurnDetector()
    is_endpoint = detector.predict(audio_array)
"""

import torch
import numpy as np
from transformers import AutoFeatureExtractor, AutoTokenizer
from .smollm_whisper import WhisperSmolLMClassifier, WhisperSmolLMClassifierConfig
import re
import logging
import os

logger = logging.getLogger(__name__)

# Disable torch compilation logs
torch._logging.set_logs(recompiles=False, cudagraphs=False)

def clean_text(txt):
    """Remove punctuation and normalize text."""
    return re.sub(r"[\.\!\,\?]", "", txt.strip().lower())


class TurnDetector:
    """
    Lightweight turn detector with torch.compile optimization.
    """
    
    def __init__(
        self,
        model_name="vogent/turn-detector",
        revision=None,
        device=None,
        compile_model=True,
        warmup=True,
    ):
        """
        Initialize the turn detector.
        
        Args:
            model_name: HuggingFace model ID (default: vogent/Vogent-Turn-80M)
            revision: Specific model revision (default: from TURN_DETECTOR_REVISION env var)
            device: 'cuda', 'cpu', or None for auto-detection
            compile_model: Whether to use torch.compile (default: True, recommended)
            warmup: Whether to warmup compiled model (default: True, recommended)
        """
        if revision is None:
            revision = os.getenv("TURN_DETECTOR_REVISION")
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set dtype based on device
        if self.device == 'cuda':
            if torch.cuda.get_device_capability()[0] >= 8:
                self._dtype = torch.bfloat16
            else:
                self._dtype = torch.float16
        else:
            self._dtype = torch.float32
        
        logger.info(f"Initializing TurnDetector on {self.device} with dtype {self._dtype}")
        
        # Load feature extractor and tokenizer
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")
        self.feature_extractor.padding_side = "left"
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
        
        # Load model config and model
        logger.info("Loading model configuration...")
        config = WhisperSmolLMClassifierConfig.from_pretrained(
            model_name,
            revision=revision,
        )
        
        logger.info(f"Loading model weights for {model_name}...")
        self.model = WhisperSmolLMClassifier.from_pretrained(
            model_name,
            config=config,
            revision=revision,
        )
        
        if self.device == 'cuda':
            self.model = self.model.to(self._dtype).cuda()
        
        self.model.eval()
        
        # Compile model for performance (optional but recommended)
        if compile_model:
            logger.info("Compiling model with torch.compile...")
            backend = "inductor" if self.device == 'cuda' else "aot_eager"
            self.model = torch.compile(
                self.model,
                mode="max-autotune",
                backend=backend
            )
            
            if warmup:
                self._warmup_model()
        
        logger.info("TurnDetector initialization complete")
    
    def _warmup_model(self):
        """
        Warmup the compiled model with common shapes.
        
        Purpose: Pre-compile kernels for expected input shapes to avoid
        latency spikes during first inference. This is critical for
        production low-latency requirements.
        """
        logger.info("Warming up model...")
        
        # Use smaller warmup sets for quick initialization
        if self.device == 'cuda':
            sequence_sizes = [64, 128, 256, 512]
            batch_sizes = [1, 2, 4]
        else:
            sequence_sizes = [64, 128]
            batch_sizes = [1]
        
        with torch.no_grad():
            for seq_size in sequence_sizes:
                for batch_size in batch_sizes:
                    # Create dummy inputs
                    tok_inputs = torch.zeros((batch_size, seq_size), dtype=torch.int64)
                    feats = torch.zeros((batch_size, 80, 800), dtype=self._dtype)
                    attn_mask = torch.ones(
                        (batch_size, tok_inputs.size(1) + feats.size(2) // 2),
                        dtype=torch.int64
                    )
                    
                    if self.device == 'cuda':
                        tok_inputs = tok_inputs.cuda()
                        feats = feats.cuda()
                        attn_mask = attn_mask.cuda()
                    
                    # Mark as dynamic for flexible shapes
                    torch._dynamo.mark_dynamic(tok_inputs, 0)
                    torch._dynamo.mark_dynamic(tok_inputs, 1)
                    torch._dynamo.mark_dynamic(feats, 0)
                    torch._dynamo.mark_dynamic(attn_mask, 0)
                    torch._dynamo.mark_dynamic(attn_mask, 1)
                    
                    # Run forward pass to trigger compilation
                    _ = self.model(
                        input_ids=tok_inputs,
                        attention_mask=attn_mask,
                        audio_features=feats,
                    ).logits
        
        logger.info("Model warmup complete")
    
    def predict(
        self,
        audio: np.ndarray,
        prev_line: str = "",
        curr_line: str = "",
        return_probs: bool = False,
        sample_rate: int = None,
    ):
        """
        Predict if the speaker has finished their turn.
        
        Args:
            audio: Audio array (mono, float32 in range [-1, 1])
                   Up to 8 seconds recommended. Longer audio will be truncated
                   to the last 8 seconds. Audio will be automatically resampled
                   to 16kHz if a different sample rate is specified.
            prev_line: Previous line of dialog (optional, for context)
            curr_line: Current line being spoken (optional, for context)
            return_probs: If True, return dict with probabilities;
                         else return boolean
            sample_rate: Sample rate of the input audio in Hz (default: None).
                        If None, assumes 16kHz. If a different rate is specified,
                        audio will be automatically resampled to 16kHz (requires librosa).
            
        Returns:
            bool: True if turn is complete (endpoint detected)
            or
            dict: {
                'is_endpoint': bool,
                'prob_endpoint': float,
                'prob_continue': float
            }
            
        Example:
            >>> detector = TurnDetector()
            >>> audio = load_audio("speech.wav")  # 16kHz, mono
            >>> result = detector.predict(
            ...     audio,
            ...     prev_line="What is your name?",
            ...     curr_line="My name is",
            ...     return_probs=True
            ... )
            >>> print(result)
            {'is_endpoint': False, 'prob_endpoint': 0.23, 'prob_continue': 0.77}
        """
        # Warn if sample rate not specified
        if sample_rate is None:
            logger.warning(
                "No sample rate specified for input audio. Assuming 16kHz. "
                "If your audio has a different sample rate, please specify it using the "
                "'sample_rate' parameter to ensure accurate turn detection. "
                "Incorrect sample rates may significantly affect performance and accuracy."
            )
            sample_rate = 16000
        
        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Resample to 16kHz if needed (model only works with 16kHz)
        target_sr = 16000
        if sample_rate != target_sr:
            try:
                import librosa
                logger.info(f"Resampling audio from {sample_rate}Hz to {target_sr}Hz...")
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
                sample_rate = target_sr
            except ImportError:
                logger.error(
                    f"librosa is required for resampling audio from {sample_rate}Hz to {target_sr}Hz. "
                    f"Install with: pip install librosa"
                )
                raise ImportError(
                    f"Cannot resample audio from {sample_rate}Hz to {target_sr}Hz without librosa. "
                    "Please install librosa: pip install librosa"
                )
        
        # Truncate to 8 seconds if needed (after resampling)
        max_samples = 8 * sample_rate
        if len(audio) > max_samples:
            audio = audio[-max_samples:]
        
        # Prepare text context using chat template
        if prev_line or curr_line:
            chat_template = self.tokenizer.apply_chat_template([
                {"role": "user", "content": clean_text(prev_line)},
                {"role": "assistant", "content": clean_text(curr_line)}
            ], tokenize=False).rstrip().rstrip("<|im_end|>")
        else:
            chat_template = ""
        
        # Tokenize text
        text_inputs = self.tokenizer(
            chat_template,
            return_tensors="pt",
            padding=True,
        )
        
        # Extract audio features (audio is now guaranteed to be 16kHz)
        audio_inputs = self.feature_extractor(
            audio,
            return_tensors="pt",
            sampling_rate=16000,
            max_length=16000 * 8,
        )
        
        # Build attention mask (audio tokens + text tokens)
        feats = audio_inputs.input_features
        if self.device == 'cuda':
            feats = feats.to(self._dtype)
        
        input_ids = text_inputs.input_ids
        attention_mask = torch.cat([
            torch.ones(
                input_ids.shape[0],
                feats.size(2) // 2,
                dtype=text_inputs.attention_mask.dtype,
                device=text_inputs.attention_mask.device
            ),
            text_inputs.attention_mask,
        ], dim=1)
        
        # Move to device
        if self.device == 'cuda':
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            feats = feats.cuda()
        
        # Run inference
        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_features=feats,
            ).logits
        
        # Get probabilities
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        prob_continue = float(probs[0])
        prob_endpoint = float(probs[1])
        
        if return_probs:
            return {
                'is_endpoint': prob_endpoint > 0.5,
                'prob_endpoint': prob_endpoint,
                'prob_continue': prob_continue,
            }
        else:
            return prob_endpoint > 0.5
    
    def predict_batch(
        self,
        audio_batch: list,
        context_batch: list = None,
        return_probs: bool = False,
        sample_rate: int = None,
    ):
        """
        Predict for multiple audio samples in a batch (more efficient).
        
        Args:
            audio_batch: List of audio arrays (each mono, float32). Audio will be
                        automatically resampled to 16kHz if a different sample rate
                        is specified.
            context_batch: List of dicts with 'prev_line' and 'curr_line' keys,
                          or None for no context
            return_probs: If True, return probabilities
            sample_rate: Sample rate of the input audio in Hz (default: None).
                        If None, assumes 16kHz. If a different rate is specified,
                        audio will be automatically resampled to 16kHz (requires librosa).
            
        Returns:
            List of predictions (bools or dicts depending on return_probs)
        """
        # Warn if sample rate not specified
        if sample_rate is None:
            logger.warning(
                "No sample rate specified for input audio. Assuming 16kHz. "
                "If your audio has a different sample rate, please specify it using the "
                "'sample_rate' parameter to ensure accurate turn detection. "
                "Incorrect sample rates may significantly affect performance and accuracy."
            )
            sample_rate = 16000
        
        if context_batch is None:
            context_batch = [{"prev_line": "", "curr_line": ""} for _ in audio_batch]
        
        # Resample to 16kHz if needed (model only works with 16kHz)
        target_sr = 16000
        if sample_rate != target_sr:
            try:
                import librosa
                logger.info(f"Resampling batch audio from {sample_rate}Hz to {target_sr}Hz...")
                audio_batch = [
                    librosa.resample(a.astype(np.float32), orig_sr=sample_rate, target_sr=target_sr)
                    for a in audio_batch
                ]
                sample_rate = target_sr
            except ImportError:
                logger.error(
                    f"librosa is required for resampling audio from {sample_rate}Hz to {target_sr}Hz. "
                    f"Install with: pip install librosa"
                )
                raise ImportError(
                    f"Cannot resample audio from {sample_rate}Hz to {target_sr}Hz without librosa. "
                    "Please install librosa: pip install librosa"
                )
        
        # Truncate audio to 8 seconds (after resampling)
        audio_batch = [
            a[-8 * sample_rate:] if len(a) > 8 * sample_rate else a
            for a in audio_batch
        ]
        
        # Pad audio to same length (left padding)
        max_len = max(len(a) for a in audio_batch)
        padded_audio = [
            np.pad(a, (max_len - len(a), 0), mode='constant')
            for a in audio_batch
        ]
        audio_array = np.vstack(padded_audio).astype(np.float32)
        
        # Prepare text contexts
        chat_templates = [
            self.tokenizer.apply_chat_template([
                {"role": "user", "content": clean_text(ctx.get("prev_line", ""))},
                {"role": "assistant", "content": clean_text(ctx.get("curr_line", ""))}
            ], tokenize=False).rstrip().rstrip("<|im_end|>")
            if ctx.get("prev_line") or ctx.get("curr_line") else ""
            for ctx in context_batch
        ]
        
        # Tokenize
        text_inputs = self.tokenizer(
            chat_templates,
            return_tensors="pt",
            padding=True,
        )
        
        # Extract features (audio is now guaranteed to be 16kHz)
        audio_inputs = self.feature_extractor(
            audio_array,
            return_tensors="pt",
            sampling_rate=16000,
            max_length=16000 * 8,
        )
        
        # Prepare inputs
        feats = audio_inputs.input_features
        if self.device == 'cuda':
            feats = feats.to(self._dtype)
        
        input_ids = text_inputs.input_ids
        attention_mask = torch.cat([
            torch.ones(
                input_ids.shape[0],
                feats.size(2) // 2,
                dtype=text_inputs.attention_mask.dtype,
            ),
            text_inputs.attention_mask,
        ], dim=1)
        
        if self.device == 'cuda':
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            feats = feats.cuda()
        
        # Run inference
        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_features=feats,
            ).logits
        
        # Get probabilities
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        results = []
        for i in range(len(audio_batch)):
            prob_continue = float(probs[i, 0])
            prob_endpoint = float(probs[i, 1])
            
            if return_probs:
                results.append({
                    'is_endpoint': prob_endpoint > 0.5,
                    'prob_endpoint': prob_endpoint,
                    'prob_continue': prob_continue,
                })
            else:
                results.append(prob_endpoint > 0.5)
        
        return results


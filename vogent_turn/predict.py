#!/usr/bin/env python3
"""
Command-line tool to run turn detection on an audio file.

Usage:
    python predict.py path/to/audio.wav --prev "What is your name?" --curr "My name is"
"""

import argparse
import numpy as np
import sys
import logging

try:
    import soundfile as sf
except ImportError:
    print("Error: soundfile is required. Install with: pip install soundfile")
    sys.exit(1)

from vogent_turn.inference import TurnDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def load_audio(file_path):
    """
    Load audio file and return audio data with its original sample rate.
    
    Note: Does NOT resample. Resampling is handled by the TurnDetector.predict() method.
    
    Returns:
        tuple: (audio, sample_rate) where audio is float32 array, mono, in range [-1, 1]
    """
    print(f"Loading audio from {file_path}...")
    audio, sr = sf.read(file_path)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        print(f"  Converting from {audio.shape[1]} channels to mono")
        audio = audio.mean(axis=1)
    
    # Normalize to float32 in [-1, 1]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    duration = len(audio) / sr
    print(f"  Loaded {duration:.2f} seconds of audio at {sr}Hz")
    
    return audio, sr

def main():
    parser = argparse.ArgumentParser(
        description='Run turn detection on an audio file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (conversational context is required)
  python predict.py speech.wav --prev "What is your name?" --curr "My name is John"
  
  # Using specific model revision
  python predict.py speech.wav --prev "Hello" --curr "Hi there" --model vogent/Vogent-Turn-80M --revision main
  
  # Use CPU instead of GPU
  python predict.py speech.wav --prev "Hello" --curr "Hi there" --device cpu

Note: Sample rate is automatically detected from the audio file.
      Audio will be resampled to 16kHz internally if needed.
        """
    )
    
    parser.add_argument(
        'audio_file',
        help='Path to audio file (wav, mp3, flac, etc.)'
    )
    parser.add_argument(
        '--prev',
        required=True,
        help='Previous line of dialog (required for context)'
    )
    parser.add_argument(
        '--curr',
        required=True,
        help='Current line being spoken (required for context)'
    )
    parser.add_argument(
        '--model',
        default='vogent/Vogent-Turn-80M',
        help='HuggingFace model name (default: vogent/Vogent-Turn-80M)'
    )
    parser.add_argument(
        '--revision',
        default=None,
        help='Model revision (default: from TURN_DETECTOR_REVISION env var)'
    )
    parser.add_argument(
        '--no-compile',
        action='store_true',
        help='Disable torch.compile (faster startup, slower inference)'
    )
    parser.add_argument(
        '--no-warmup',
        action='store_true',
        help='Skip model warmup (faster startup, may have latency spike on first use)'
    )
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Initialize detector (this will take time on first run)
    print("\n" + "="*70)
    print("Initializing Turn Detector")
    print("="*70)
    
    device = None if args.device == 'auto' else args.device
    
    detector = TurnDetector(
        model_name=args.model,
        revision=args.revision,
        device=device,
        compile_model=not args.no_compile,
        warmup=not args.no_warmup,
    )
    
    print("\n" + "="*70)
    print("Running Turn Detection")
    print("="*70)
    
    # Process the audio file
    try:
        # Load audio (returns audio and its original sample rate)
        audio, sample_rate = load_audio(args.audio_file)
        
        # Run prediction
        print(f"\nContext:")
        print(f"  Previous: '{args.prev}'")
        print(f"  Current: '{args.curr}'")
        
        print(f"\nAnalyzing (audio will be resampled to 16kHz if needed)...")
        result = detector.predict(
            audio,
            prev_line=args.prev,
            curr_line=args.curr,
            return_probs=True,
            sample_rate=sample_rate,  # Pass detected sample rate from file
        )
        
        # Display results
        print("\n" + "-"*70)
        print(f"File: {args.audio_file}")
        print("-"*70)
        
        if result['is_endpoint']:
            print("✅ TURN COMPLETE")
        else:
            print("⏸️  TURN INCOMPLETE (speaker may continue)")
        
        print(f"\nConfidence: {max(result['prob_endpoint'], result['prob_continue']):.1%}")
        print(f"  Probability turn complete: {result['prob_endpoint']:.1%}")
        print(f"  Probability continuing:    {result['prob_continue']:.1%}")
        print("-"*70)
            
    except FileNotFoundError:
        print(f"\n❌ Error: File not found: {args.audio_file}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error processing {args.audio_file}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*70)
    print("Done")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()


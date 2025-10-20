#!/usr/bin/env python3
"""
Basic usage example for vogent-turn.

This example shows the simplest way to use the turn detector
with a single audio file and conversational context.
"""

from vogent_turn import TurnDetector
import soundfile as sf
import sys

def main():
    print("Initializing Turn Detector...")
    detector = TurnDetector(
        compile_model=True,  # Enable torch.compile for faster inference
        warmup=True,         # Warmup common shapes
    )
    
    # Download audio file from a URL (replace with actual URL later)
    import urllib.request
    audio_url = "https://storage.googleapis.com/voturn-sample-recordings/incomplete_number_sample.wav"  # TODO: Replace with actual hosted .wav URL
    audio_file = "incomplete_number_sample.wav"
    print(f"\nDownloading audio from {audio_url} ...")
    urllib.request.urlretrieve(audio_url, audio_file)
    print(f"Saved to {audio_file}")
    
    try:
        audio, sr = sf.read(audio_file)
    except FileNotFoundError:
        print(f"Error: {audio_file} not found!")
        print("Please provide a valid audio file or create a test file.")
        sys.exit(1)
    
    # Define conversational context
    prev_line = "What is your phone number"
    curr_line = "My number is 804"
    
    print(f"\nConversational context:")
    print(f"  Previous: '{prev_line}'")
    print(f"  Current: '{curr_line}'")
    
    # Run turn detection
    print("\nAnalyzing turn endpoint...")
    result = detector.predict(
        audio,
        prev_line=prev_line,
        curr_line=curr_line,
        sample_rate=sr,
        return_probs=True,
    )
    
    # Display results
    print("\n" + "="*60)
    if result['is_endpoint']:
        print("✅ TURN COMPLETE")
        print("The speaker has finished their turn.")
    else:
        print("⏸️  TURN INCOMPLETE")
        print("The speaker may continue speaking.")
    
    print(f"\nConfidence scores:")
    print(f"  Turn complete:   {result['prob_endpoint']:.1%}")
    print(f"  Turn continuing: {result['prob_continue']:.1%}")
    print("="*60)

if __name__ == "__main__":
    main()


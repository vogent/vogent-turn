#!/usr/bin/env python3
"""
Batch processing example for vogent-turn.

This example shows how to efficiently process multiple audio files using
predict_batch() with unique conversational context for each file.

Audio files are downloaded from URLs and processed in a single batch for
optimal performance.
"""

from vogent_turn import TurnDetector
import soundfile as sf
import requests
import tempfile
import os

def download_audio(url, filename):
    """Download audio file from URL to temporary location."""
    print(f"  Downloading: {filename}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    temp_path = os.path.join(tempfile.gettempdir(), filename)
    with open(temp_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return temp_path

def main():
    print("Initializing Turn Detector...")
    detector = TurnDetector()
    
    # TODO: Populate these URLs with your audio file locations
    audio_sources = [
        {
            "url": "https://storage.googleapis.com/voturn-sample-recordings/incomplete_number_sample.wav",  # TODO: Replace with actual URL
            "filename": "incomplete_number_sample.wav",
            "prev_line": "What is your phone number",
            "curr_line": "My number is 804"
        },
        {
            "url": "https://storage.googleapis.com/voturn-sample-recordings/complete_number_sample.wav",  # TODO: Replace with actual URL
            "filename": "complete_number_sample.wav",
            "prev_line": "What is your phone number",
            "curr_line": "My number is 8042221111"
        },
    ]
    
    print(f"\nDownloading {len(audio_sources)} audio files...\n")
    
    # Download and load all audio files
    audio_batch = []
    context_batch = []
    filenames = []
    temp_files = []
    sample_rate = None
    
    for source in audio_sources:
        try:
            # Download audio file
            temp_path = download_audio(source["url"], source["filename"])
            temp_files.append(temp_path)
            
            # Load audio
            audio, sr = sf.read(temp_path)
            
            # Store sample rate (should be same for all files in batch)
            if sample_rate is None:
                sample_rate = sr
            elif sample_rate != sr:
                print(f"  ⚠️  Warning: {source['filename']} has different sample rate ({sr} vs {sample_rate})")
            
            audio_batch.append(audio)
            context_batch.append({
                "prev_line": source["prev_line"],
                "curr_line": source["curr_line"]
            })
            filenames.append(source["filename"])
            
            print(f"  ✓ Loaded: {source['filename']} ({sr} Hz)")
            
        except Exception as e:
            print(f"  ❌ Error loading {source['filename']}: {e}")
            continue
    
    if not audio_batch:
        print("\n❌ No audio files loaded successfully. Exiting.")
        return
    
    # Process all audio files in a single efficient batch
    print(f"\nProcessing batch of {len(audio_batch)} files...")
    try:
        results = detector.predict_batch(
            audio_batch,
            context_batch=context_batch,
            sample_rate=sample_rate,
            return_probs=True,
        )
        print("✓ Batch processing complete!\n")
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        return
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
    
    # Print summary
    print("="*70)
    print("BATCH PROCESSING RESULTS")
    print("="*70)
    for i, result in enumerate(results):
        status = "✅ ENDPOINT" if result['is_endpoint'] else "⏸️  CONTINUE"
        print(f"\n{status} - {filenames[i]}")
        print(f"  Confidence: {result['prob_endpoint']:.1%}")
        print(f"  Context: {context_batch[i]['prev_line']} → {context_batch[i]['curr_line']}")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()


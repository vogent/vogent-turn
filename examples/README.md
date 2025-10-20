# Vogent Turn Sample Scripts 

This folder contains scripts for running Vogent Turn on a couple of audio samples: [an incomplete phone number](https://storage.googleapis.com/voturn-sample-recordings/incomplete_number_sample.wav) and [a complete phone number](https://storage.googleapis.com/voturn-sample-recordings/complete_number_sample.wav)

`python basic_usage.py` runs the turn detector on the audio sample of an incomplete phone number reading.

`python batch_processing.py` runs the turn detector on the both samples as a batched input. 

`request_batcher.py` is example code for a thread that can continuously receive and batch requests. This is useful for productionizing the turn detector in e.g. a voice conversation setting.

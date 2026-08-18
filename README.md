#CodeAlpha_AI Music Generator 🎵

This is a Deep Learning project that uses a Recurrent Neural Network (LSTM) to generate completely new music sequences. 

## How it Works
* Parses MIDI files to extract musical notes and chords using `music21`.
* Trains an LSTM Neural Network on these sequences.
* Generates a new array of notes and converts it back into a playable `.mid` audio file.

## Tech Stack
* **Language:** Python
* **Libraries:** TensorFlow, Keras, NumPy, Music21

## How to Run
1. Run `python train.py` to train the model on your MIDI files.
2. Run `python generate.py` to generate a new AI track.

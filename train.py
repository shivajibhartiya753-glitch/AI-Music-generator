import glob
import numpy as np
from music21 import converter, instrument, note, chord
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation
from tensorflow.keras.utils import to_categorical

print("1. Loading MIDI files...")
notes = []

# Smart logic to read MIDI files
for file in glob.glob("midi_songs/*.mid"):
    print(f"\nReading {file}...")
    try:
        midi = converter.parse(file)
        parts = instrument.partitionByInstrument(midi)
        notes_to_parse = []
        
        if parts: # If the song has multiple instruments, extract from all
            for part in parts.parts:
                notes_to_parse.extend(part.recurse())
        else: # If it is a single track
            notes_to_parse = midi.flat.notes
            
        file_notes = 0
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
                file_notes += 1
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                file_notes += 1
                
        print(f" -> Success! {file_notes} notes/chords extracted.")
    except Exception as e:
        print(f" -> Error: Could not read this file. ({e})")

if not notes:
    print("\nERROR: Could not extract any notes from the MIDI files.")
    print("Solution: These files might be empty or in an unsupported format. Please try downloading new .mid files.")
    exit()

print(f"\nTotal notes parsed: {len(notes)}")
print("\n2. Preparing data for the Neural Network...")

sequence_length = 50 
pitches = sorted(set(item for item in notes))
n_vocab = len(pitches)
note_to_int = dict((note, number) for number, note in enumerate(pitches))

network_input = []
network_output = []

for i in range(0, len(notes) - sequence_length, 1):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]
    network_input.append([note_to_int[char] for char in sequence_in])
    network_output.append(note_to_int[sequence_out])

n_patterns = len(network_input)
network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
network_input = network_input / float(n_vocab)
network_output = to_categorical(network_output)

print("3. Building the LSTM Model...")
model = Sequential()
model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2])))
model.add(Dropout(0.3))
model.add(Dense(n_vocab))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print("4. Training started... (This may take some time depending on hardware)")
model.fit(network_input, network_output, epochs=10, batch_size=64)

model.save("music_model.keras")
print("\nModel successfully trained and saved! 🎉")
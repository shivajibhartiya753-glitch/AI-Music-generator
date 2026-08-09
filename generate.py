import glob
import numpy as np
import random
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import load_model

print("1. Loading data to map notes for the model...")
notes = []
for file in glob.glob("midi_songs/*.mid"):
    try:
        midi = converter.parse(file)
        parts = instrument.partitionByInstrument(midi)
        notes_to_parse = []
        
        if parts:
            for part in parts.parts:
                notes_to_parse.extend(part.recurse())
        else:
            notes_to_parse = midi.flat.notes
            
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    except:
        pass

pitches = sorted(set(item for item in notes))
n_vocab = len(pitches)
note_to_int = dict((note, number) for number, note in enumerate(pitches))
int_to_note = dict((number, note) for number, note in enumerate(pitches))

sequence_length = 50
network_input = []
for i in range(0, len(notes) - sequence_length, 1):
    sequence_in = notes[i:i + sequence_length]
    network_input.append([note_to_int[char] for char in sequence_in])

print("2. Loading your trained AI model...")
model = load_model("music_model.keras")

print("3. Generating new music (This will take a few seconds)...")
start = random.randint(0, len(network_input)-1)
pattern = network_input[start]
prediction_output = []

# Asking the AI to predict 100 new notes
for note_index in range(1000):
    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction_input = prediction_input / float(n_vocab)
    
    prediction = model.predict(prediction_input, verbose=0)
    index = np.argmax(prediction)
    result = int_to_note[index]
    prediction_output.append(result)
    
    pattern.append(index)
    pattern = pattern[1:len(pattern)]

print("4. Converting AI-generated notes into a MIDI audio file...")
offset = 0
output_notes = []

for pattern in prediction_output:
    if ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))
            new_note.storedInstrument = instrument.Piano()
            notes.append(new_note)
        new_chord = chord.Chord(notes)
        new_chord.offset = offset
        output_notes.append(new_chord)
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)
    
    offset += 0.5

midi_stream = stream.Stream(output_notes)
# The new music file will be saved with this name
midi_stream.write('midi', fp='AI_Mera_Gaana.mid')

print("🎉 DONE! A new file named 'AI_Mera_Gaana.mid' has been created. Go listen to it!")
from music21 import converter, instrument, note, chord
import os
import numpy

notes = []

# get all midi files simultaneously
training_folder = 'midi_data/'
midi_files = [file for file in os.listdir(training_folder) if os.path.isfile(os.path.join(training_folder, file)) and os.path.splitext(file)[1] == '.mid']

for midi_file in midi_files:
    midi = converter.parse(training_folder + midi_file)
    midi_notes = None

    parts = instrument.partitionByInstrument(midi)

    if parts:
        midi_notes = parts.parts[0].recurse()
    else:
        midi_notes = midi.flat.notes

    for element in midi_notes:
        if(isinstance(element, note.Note)):
            notes.append(str(element.pitch))
        elif(isinstance(element, chord.Chord)):
            notes.append('.'.join(str(n) for n in element.normalOrder))


seq_len = 100

# map pitches to numbers
pitch_names = sorted(set(item for item in notes))
note_to_int = dict((note, number) for number, note in enumerate(pitch_names))

network_input = []
network_output = []

for i in range(0, len(notes) - seq_len, 1):
    seq_in = notes[i: i + seq_len]
    seq_out = notes[i + seq_len]
    network_input.append([note_to_int[char] for char in seq_in])
    network_output.append(note_to_int[seq_out])

num_patterns = len(network_input)

# reshape
network_input = numpy.reshape(network_input, (num_patterns, seq_len, 1))
# normalize
network_input = network_input / float(len(set(notes)))

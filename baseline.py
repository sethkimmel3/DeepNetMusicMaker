"""
GENERATE ARTIFICIAL MUSIC BASELINE METHOD
Code mostly taken from: https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
"""

from music21 import converter, instrument, note, chord, stream
import os
import numpy
from keras.utils import np_utils
import keras
from keras.callbacks import ModelCheckpoint
import pickle

def getNotes():
    notes = []
    # out_nodes = []
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
                #out_nodes.append(element)
            elif(isinstance(element, chord.Chord)):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                #out_nodes.extend(element.notes)
            # if(len(out_nodes) > 100):
            #     get_midi_sequence(out_nodes)
            #     break

    # with open('data/notes', 'wb') as filepath:
    #     pickle.dump(notes, filepath)

    a_vocab = len(set(notes))
    return notes,a_vocab

# def get_midi_sequence(notes):
#     midi_stream = stream.Stream(notes)
#     midi_stream.write('midi', fp='example_input.mid')

def prepare_sequences(notes, a_vocab):
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
    network_input = network_input / float(a_vocab)

    network_output = np_utils.to_categorical(network_output)

    return network_input,network_output,pitch_names

def prepare_generative_sequences(notes, a_vocab):

    pitch_names = sorted(set(item for item in notes))

    note_to_int = dict((note, number) for number, note in enumerate(pitch_names))

    seq_len = 100
    network_input = []
    output = []

    for i in range(0, len(notes) - seq_len, 1):
        seq_in = notes[i:i + seq_len]
        seq_out = notes[i + seq_len]
        network_input.append([note_to_int[char] for char in seq_in])
        output.append((note_to_int[seq_out]))

    n_patterns = len(network_input)

    normalized_input = numpy.reshape(network_input, (n_patterns, seq_len, 1))

    normalized_input = normalized_input / float(a_vocab)

    return network_input,normalized_input,pitch_names


def getModel(network_input, a_vocab):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(512, input_shape=(network_input.shape[1],network_input.shape[2]), recurrent_dropout=0.3, return_sequences=True))
    model.add(keras.layers.LSTM(512, return_sequences=True, recurrent_dropout=0.3))
    model.add(keras.layers.LSTM(512))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(a_vocab))
    model.add(keras.layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

    return model

def getGenerativeModel(network_input, a_vocab):
    model = getModel(network_input, a_vocab)
    model.load_weights("weights.hdf5")
    return model


def train(model, network_input, network_output):
    filepath = "weights.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=5, batch_size=128, callbacks=callbacks_list)

def generateNotes(model, network_input, pitch_names, a_vocab):
    # pick a random starting point
    start = numpy.random.randint(0, len(network_input) - 1)

    int_to_note = dict((number,note) for number, note in enumerate(pitch_names))

    pattern = network_input[start]

    prediction_seq = []

    # generate n notes
    n = 500
    for note_index in range(n):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(a_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_seq.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_seq

def create_midi(prediction_seq):
    offset = 0
    output_notes = []

    for pattern in prediction_seq:
        # pattern is a chord
        if('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase the offset so notes don't overlap
        offset += 0.5

        midi_stream = stream.Stream(output_notes)

        midi_stream.write('midi', fp='test_output.mid')

# train the network
def train_network():
    notes,a_vocab = getNotes()
    network_input,network_output,pitch_names = prepare_sequences(notes,a_vocab)
    model = getModel(network_input, a_vocab)
    train(model, network_input, network_output)

# create a new song from the network
def generate_new_midi():
    notes,a_vocab = getNotes()
    network_input, normalized_input, pitch_names = prepare_generative_sequences(notes, a_vocab)
    generativeModel = getGenerativeModel(normalized_input, a_vocab)
    prediction_seq = generateNotes(generativeModel, network_input, pitch_names, a_vocab)
    create_midi(prediction_seq)

train_network()
generate_new_midi()
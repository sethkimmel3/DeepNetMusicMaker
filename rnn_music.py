# from google.colab import drive
# import os
# drive.mount('/content/drive')
# os.chdir('/content/drive/My Drive/')

import pickle
import keras
import numpy as np
from music21 import *
import matplotlib.pyplot as plt


all_notes = pickle.load(open("notes_all.pickle","rb")) # open the file with all notes

one_hot_offsets = np.zeros((len(all_notes), 9)) # 9 dim for onset
one_hot_durations = np.zeros((len(all_notes), 9)) # 9 dim for duration
one_hot_pitches = np.zeros((len(all_notes), 12)) # 12 dim for pitches
for i in range(len(all_notes)): # go through notes and encode onset, duration, and pitch
	n = all_notes[i]
	one_hot_offsets[i,n[0]] = 1 
	one_hot_durations[i,n[1]] = 1
	one_hot_pitches[i,n[2]] = 1

timesteps = 10 # timesteps necessary for RNN, i.e. the RNN sees this many notes back when coming up with the next note
num_notes = len(all_notes) # number of notes in total
num_offsets = one_hot_offsets.shape[1]
num_durations = one_hot_durations.shape[1]
num_pitches = one_hot_pitches.shape[1]

# x contains the inputs to the network
x_offsets = np.zeros((num_notes-timesteps, timesteps, num_offsets)).astype(np.bool)
x_durations = np.zeros((num_notes-timesteps, timesteps, num_durations)).astype(np.bool)
x_pitches = np.zeros((num_notes-timesteps, timesteps, num_pitches)).astype(np.bool)
for t in range(timesteps): # x contains the past timesteps necessary to predict the next timestep
	x_offsets[:,t,:] = one_hot_offsets[t:t-timesteps,:].astype(np.bool)
	x_durations[:,t,:] = one_hot_durations[t:t-timesteps,:].astype(np.bool)
	x_pitches[:,t,:] = one_hot_pitches[t:t-timesteps,:].astype(np.bool)
print(x_offsets.shape) # has shape (num_notes, timesteps, num_offsets)

# y contains the future notes that the RNN needs to predict
y_offsets = one_hot_offsets[timesteps:,:].astype(np.bool)
y_durations = one_hot_durations[timesteps:,:].astype(np.bool)
y_pitches = one_hot_pitches[timesteps:,:].astype(np.bool)
print(y_offsets.shape)

pitches_in = keras.layers.Input(shape=(timesteps, num_pitches,)) # inputs from the notes
pitches_out = keras.layers.LSTM(128)(pitches_in) # output uses an LSTM and then a Dense layer
pitches_out = keras.layers.Dense(num_pitches, activation='softmax')(pitches_out)

offsets_in = keras.layers.Input(shape=(timesteps, num_offsets,)) # inputs from the offsets
offsets_out = keras.layers.LSTM(128)(offsets_in) # output uses an LSTM and then a Dense layer
offsets_out = keras.layers.Dense(num_offsets, activation='softmax')(offsets_out)

durations_in = keras.layers.Input(shape=(timesteps, num_durations,)) # inputs from the pitches
durations_out = keras.layers.LSTM(128)(durations_in) # output uses an LSTM and then a Dense layer
durations_out = keras.layers.Dense(num_durations, activation='softmax')(durations_out)

model = keras.Model(inputs=[offsets_in, durations_in, pitches_in], outputs=[offsets_out, durations_out, pitches_out])
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit([x_offsets, x_durations, x_pitches], [y_offsets, y_durations, y_pitches], epochs=5) # run for 5 epochs
model.save('trained_2.h5')

# use notes to turn into midi file, start at a random point (-1) or at a given point
def convertToMIDI(all_notes, length, start=-1):
	total_offset = 0 # total time within the stream
	curr_stream = stream.Stream() # make a stream to hold notes
	if start == -1:
		start = np.random.randint(0, len(all_notes)-length) # choose a random starting point
	for t in range(length):
		n = all_notes[t+start] # get the note
		offset = n[0] # find offset, duration, and pitch
		dur = n[1]
		ps = n[2]
		curr_note = note.Note() # make a note
		curr_note.duration = duration.Duration(dur/4) # set to correct duration (in quarter notes)
		curr_note.pitch.ps = ps+60 # pitch is relative to middle C = 60
		curr_stream.insert(offset/4+total_offset, curr_note) # insert at the correct offset (in quarter notes)
		total_offset += offset/4 # move to next time
	curr_stream.write('midi','out_music_true.mid') # write out the file


# use probabilities to randomly choose a value within the array
def probChoose(array):
	num = np.random.rand()
	for i in range(len(array)):
		if num < array[i]:
			return i
		else:
			num -= array[i]


# generate a midi file starting with some input and going on for length
def generate(model, starting_inputs, length):
	offsets, durations, pitches = starting_inputs
	offsets = np.reshape(offsets, (1,offsets.shape[0], offsets.shape[1])) # inputs to pass into the network
	durations = np.reshape(durations, (1,durations.shape[0], durations.shape[1]))
	pitches = np.reshape(pitches, (1,pitches.shape[0], pitches.shape[1]))
	total_offset = 0
	curr_stream = stream.Stream()
	for t in range(length):
		next_inputs = model.predict([offsets, durations, pitches]) # predict the next input
		offset = probChoose(next_inputs[0][0,:]) # choose a random element with probabilities given by the array value
		dur = probChoose(next_inputs[1][0,:])
		ps = probChoose(next_inputs[2][0,:])
		curr_note = note.Note() # make a note
		curr_note.duration = duration.Duration(dur/4) # set to correct duration
		curr_note.pitch.ps = ps+60 # pitch the note correctly
		curr_stream.insert(offset/4+total_offset, curr_note) # insert into stream at the correct offset
		total_offset += offset/4 # move to next total offset (since offset gives the time between notes)
		offsets[0,:-1,:] = offsets[0,1:,:] # shift over inputs
		durations[0,:-1,:] = durations[0,1:,:] # shift over inputs
		pitches[0,:-1,:] = pitches[0,1:,:] # shift over inputs
		offsets[0,-1,:] = 0 # add the generated input
		offsets[0,-1,offset] = 1
		durations[0,-1,:] = 0 # add the generated input
		durations[0,-1,dur] = 1
		pitches[0,-1,:] = 0 # add the generated input
		pitches[0,-1,ps] = 1
	curr_stream.write('midi','out_music.mid')
		


model = keras.models.load_model('trained_1.h5')
random_start = np.random.randint(0,num_notes-timesteps)
convertToMIDI(all_notes, 1000, start=random_start)
starting_inputs = (one_hot_offsets[random_start:random_start+timesteps,:],
				one_hot_durations[random_start:random_start+timesteps,:],
				one_hot_pitches[random_start:random_start+timesteps,:])
generate(model, starting_inputs, 1000)

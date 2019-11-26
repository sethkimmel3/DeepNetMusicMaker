import pickle
import keras
import numpy as np



all_notes = pickle.load(open("notes_all.pickle","rb")) # open the file with all notes
one_hot_encoded_notes = np.zeros((len(all_notes), 9+9+12)) # 9 dim for onset, 9 dim for duration, 12 dim for notes
for i in range(len(all_notes)): # go through notes and encode onset, duration, and pitch
	n = all_notes[i]
	one_hot_encoded_notes[i,n[0]] = 1 
	one_hot_encoded_notes[i,9+n[1]] = 1
	one_hot_encoded_notes[i,18+n[2]] = 1
print(one_hot_encoded_notes.shape)

timesteps = 10 # timesteps necessary for RNN, i.e. the RNN sees this many notes back when coming up with the next note
num_notes = one_hot_encoded_notes.shape[0] # number of notes in the dataset
data_dim = one_hot_encoded_notes.shape[1] # dimension of the dataset (should be 30)

x = [] 
for t in range(timesteps): # x contains the past timesteps necessary to predict the next timestep
	x.append(one_hot_encoded_notes[t:t-timesteps,:])
x = np.array(x)
x = np.transpose(x, axes=(1,0,2)).astype(np.bool) # switch the first two axes so that is has shape (num_notes, timesteps, data_dim)
print(x.shape)

y = one_hot_encoded_notes[timesteps:,:].astype(np.bool) # y contains the future notes that the RNN needs to predict
print(y.shape)


model = keras.Sequential() # create a model
model.add(keras.layers.LSTM(128, input_shape=(None,30))) # LSTM takes in all 30 notes
model.add(keras.layers.Dense(30, activation='softmax')) # output goes through softmax
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(x, y, epochs=5) # run for 5 epochs

import pickle
import keras
import numpy as np


all_notes = pickle.load(open("notes_all.pickle","rb"))
one_hot_encoded_notes = np.zeros((len(all_notes), 9+9+12)) # 9 dim for onset, 9 dim for duration, 12 dim for notes
for i in range(len(all_notes)):
	n = all_notes[i]
	one_hot_encoded_notes[i,n[0]] = 1
	one_hot_encoded_notes[i,9+n[1]] = 1
	one_hot_encoded_notes[i,18+n[2]] = 1
print(one_hot_encoded_notes.shape)

timesteps = 10
num_notes = one_hot_encoded_notes.shape[0]
data_dim = one_hot_encoded_notes.shape[1]

x = []
for t in range(timesteps): # this needs to be 3D!! (num_notes, timesteps, data_dim)
	x.append(one_hot_encoded_notes[t:t-timesteps,:])
x = np.array(x)
x = np.transpose(x, axes=(1,0,2)).astype(np.bool)
print(x.shape)

y = one_hot_encoded_notes[timesteps:,:].astype(np.bool)
print(y.shape)


model = keras.Sequential()
model.add(keras.layers.LSTM(128, input_shape=(None,30)))
model.add(keras.layers.Dense(30, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(x, y, epochs=5)
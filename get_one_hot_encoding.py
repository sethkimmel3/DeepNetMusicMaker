## all_notes gives a list of tuples, (onset of note relative to previous in 16th notes, duration of note in 16th notes, pitch as 0->11 (C->B))
## the onset and duration both go from 0->8, and pitch goes from 0->11 in value
all_notes = pickle.load(open("notes_all.pickle","rb"))
one_hot_encoded_notes = np.zeros((len(all_notes), 9+9+12)) # 9 dim for onset, 9 dim for duration, 12 dim for notes
for i in range(len(all_notes)):
	n = all_notes[i]
	one_hot_encoded_notes[i,n[0]] = 1
	one_hot_encoded_notes[i,9+n[1]] = 1
	one_hot_encoded_notes[i,18+n[2]] = 1
print(one_hot_encoded_notes.shape)

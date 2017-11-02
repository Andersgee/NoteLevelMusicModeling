import midi
import numpy as np

A=np.load("test.npy")

pattern = midi.Pattern()
track = midi.Track()
pattern.append(track)

for n in range(1, A.shape[0]):
	tickdelta=(A[n,0]-A[n-1,0])*20  # (20 = 480/24)
	if A[n,1] < 128:
		track.append(midi.NoteOnEvent(tick=tickdelta, velocity=127, pitch=A[n,1]))
	else:
		track.append(midi.NoteOnEvent(tick=tickdelta, velocity=0, pitch=A[n,1]-128))

track.append(midi.EndOfTrackEvent(tick=1))

midi.write_midifile("example.mid", pattern)
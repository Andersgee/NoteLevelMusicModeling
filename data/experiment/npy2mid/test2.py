import midi
import numpy as np

fn="generated5"

A=np.load(fn+".npy")

pattern = midi.Pattern()
track = midi.Track()
pattern.append(track)

tickdelta=0
for x in range(A.shape[1]):
	tickdelta += 20 # (20 = 480/24)
	for y in range(256):
		if A[y,x] > 0:
			if y < 128:
				track.append(midi.NoteOnEvent(tick=tickdelta, velocity=int(A[y,x]*127), pitch=y))
				tickdelta=0
			else:
				track.append(midi.NoteOnEvent(tick=tickdelta, velocity=0, pitch=y-128))
				tickdelta=0

track.append(midi.EndOfTrackEvent(tick=1))

midi.write_midifile(fn+".mid", pattern)
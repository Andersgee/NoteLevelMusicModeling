from midiutil.MidiFile import MIDIFile
import numpy as np

filename="generated2"
A = np.load(filename+".npy")

mf = MIDIFile(numTracks=1, removeDuplicates=True, deinterleave=True, adjust_origin=True)
mf.addTrackName(0, 0, filename)
clocks_per_tick=20
mf.addTimeSignature(0, 0, 24, 4, clocks_per_tick, notes_per_quarter=8)
bpm=6*60 # 6 4th-notes/second (means 12 8th/s means 24 16th/s)
mf.addTempo(0, 0, bpm)

for x in range(A.shape[1]-48):
	for y in range(128):
		if A[y,x]>0:
			duration=2 # 2 would mean default to 8th notes (the unit is 16th note.. so 2 is an 8th note)
			for n in range(1,48):
				if A[y+128,x+n]>0:
					duration=n
					break
			volume = max(A[y,x]*127, 0.2*127) # make sure notes are audible
			mf.addNote(0, 0, y, x/4.0, duration/4.0, volume)
			#mf.addNote(track, channel, pitch, time, duration, volume)

with open(filename+".mid", 'wb') as fn:
	mf.writeFile(fn)
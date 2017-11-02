from midiutil.MidiFile import MIDIFile
import numpy as np

filename="generated6"
A = np.load(filename+".npy")

mf = MIDIFile(numTracks=1, removeDuplicates=True, deinterleave=True, adjust_origin=True)
mf.addTrackName(0, 0, filename)

#clocks_per_tick=24/4 * 24/4 #metronome timing 4 clicks per bar.
# 24 * X   means click every Xth 4thnote (definition)
# 24/2 * X   means click every Xth 8thnote
# 24/4 * 24/4   (pure coincidence) means click every 6th 16thnote, which is 4 times per bar
# one could have thought that this was related to midi ticks, but no.
# turns out this doesnt affect anything relevant whatsoever.
clocks_per_tick=24

numerator = 6 #how many notes to count per bar
denominator = 2 # 4th notes (the kind of note the numerator refers to. x would mean 2^x th notes)
# 6/4th per bar is equivalent to 24/16th per bar which is what I really have.

mf.addTimeSignature(0, 0, numerator, denominator, clocks_per_tick, notes_per_quarter=8)
bpm=6*60 # beats per minute aka quarternotes per minute. 6 4th-notes/second means 12 8th/s means 24 16th/s
mf.addTempo(0, 0, bpm)

defaultduration = 4 # default note duration (in 16th notes, which there are 24 of per bar)
maxduration = 2 # maximum note duration (in bars. A bar is exactly one second)
minvolume = 0.2 # [0...1]
for x in range(A.shape[1]-24*maxduration):
	for y in range(128):
		if A[y,x]>0:
			duration = defaultduration
			for n in range(1,24*maxduration):
				if A[y+128,x+n]>0:
					duration=n
					break
			volume = max(A[y,x]*127, minvolume*127)
			mf.addNote(0, 0, y, x/4.0, duration/4.0, volume)
			#mf.addNote(track, channel, pitch, time, duration, volume)

with open(filename+".mid", 'wb') as fn:
	mf.writeFile(fn)
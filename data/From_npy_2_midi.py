from midiutil.MidiFile import MIDIFile
import numpy as np
import argparse

def convert(filename):
	# Note to self:
	# Since the generated data is in steps of 1/24 second intervals, Im choosing
	# to deal with units in terms of 24 notes per bar to make a bar be 1 second always.
	# Since music notation is the way it is, I have to choose what a 1/24 second note LOOKS like.
	# Lets choose a 16th note (the one with 2 "hooks" on it), and make each bar contain 24 16th notes.
	# That means a time signature 24/16 which is equivalent to 6/4.
	#
	# Also; the last argument of addTimeSignature() concerns metronome clicks, which is irrellevant but not optional.
	#     24 simply means click every quarternote and has nothing to do with the other "24" integers in this function.
	# Also2; a 16th note has 1/4 the length of a 4th note obviously, which is why I divide by 4 here and there
	#     because the functions want 4th notes as arguments.
	#
	# midiutil.MidiFile writes format1 files, which has resolution ("Divisions") 960 instead of 480 like format0 files. so divide tempo by 2.
	# Actually, ignore the line above.

	C=1 #scaledown factor

	generated = np.load("generated_npy/"+filename+".npy")
	mf = MIDIFile(numTracks=1, removeDuplicates=True, deinterleave=True, adjust_origin=True)
	mf.addTrackName(0, 0, filename)
	#mf.addTimeSignature(0, 0, 24, 4, 24) # 24 16th notes per bar (16=2^4)	
	mf.addTimeSignature(0, 0, 6, 2, 24) # 6 4th notes per bar (4=2^2)

	#mf.addTempo(0, 0, 24/4*60) # 24 16th notes per second (this is the tempo its supposed to be)
	#mf.addTempo(0, 0, 24/4*60/2) # generally sound better at half speed...
	#mf.addTempo(0, 0, 24/4*60/4) # or even a quarter spead.

	mf.addTempo(0, 0, 24/4*60/C)

	defaultduration = 4 # in 16th notes (there are 24 per bar, which is 1 second)
	#maxduration = 2*24 # in 16th notes (2*24 means 2 seconds)
	maxduration = 2*24/C
	minvolume = 0.15 # [0...1]
	T=0.0
	#T=0.2

	for x in range(generated.shape[1]-maxduration):
		for y in range(128):
			if generated[y,x]>T:
				duration = defaultduration
				for n in range(1,maxduration):
					if generated[y+128,x+n]>T:
						duration=n
						break
				volume = max(minvolume, generated[y,x])
				mf.addNote(0, 0, y, x/4.0, duration/4.0, volume*127)

	with open("generated_mid/"+filename+".mid", 'wb') as fn:
		mf.writeFile(fn)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Converts [256,X] numpy array to a midi file and stores it in generated_mid/filename.mid")
	parser.add_argument('filename', action="store", help='as in generated_npy/filename.npy')
	args = parser.parse_args()
	convert(args.filename)
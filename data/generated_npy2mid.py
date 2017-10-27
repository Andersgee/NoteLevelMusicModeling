from midiutil.MidiFile import MIDIFile
import numpy as np
import argparse


def convert(filename):
	generated = np.load("generated_npy/"+filename+".npy")

	mf = MIDIFile(numTracks=1, removeDuplicates=True, deinterleave=True, adjust_origin=True)
	#mf = MIDIFile(1) #gives warning
	mf.addTrackName(0, 0, filename)
	bpm=250
	mf.addTempo(0, 0, bpm)

	#experiment with this aswell as volume
	#T_on = 0.6
	#T_off = 0.5
	#T_on = 0.35
	T_on = 0.0
	T_off = 0.0
	for x in range(generated.shape[1]-4):
		for n in range(128):
			if generated[n,x]>T_on:
				#search forward a bit to know note duration.
				if generated[n+128,x+4]>T_off:
					duration=4
				elif generated[n+128,x+3]>T_off:
					duration=3
				elif generated[n+128,x+2]>T_off:
					duration=2
				elif generated[n+128,x+1]>T_off:
					duration=1
				else:
					duration=1
				volume = generated[n,x] * 127
				mf.addNote(0, 0, n, x, duration, volume)
				#mf.addNote(track, channel, pitch, time, duration, volume)

	with open("generated_mid/"+filename+".mid", 'wb') as fn:
		mf.writeFile(fn)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Converts [256,X] numpy array to a midi file and stores it in generated_mid/filename.mid")
	parser.add_argument('filename', action="store", help='as in generated_npy/filename.npy')
	args = parser.parse_args()
	convert(args.filename)
	
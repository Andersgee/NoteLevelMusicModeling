import midi
import numpy as np
import argparse

def convert(filename):
	track = midi.read_midifile("mid/"+filename+".mid")[0]

	# default values
	TicksPerQuarterNote = 480 # "Resolution" (480 seems to be standard in format0 files)
	MicroSecondsPerQuarterNote = 500000 # "Tempo" #120 quarternotes per minute is 1/(120/60/1000000)

	AbsoluteMicroSeconds = 0
	note=0
	A = np.zeros([1,2], dtype=np.uint32)

	for event in track:
		Microseconds = MicroSecondsPerQuarterNote*event.tick/TicksPerQuarterNote
		AbsoluteMicroSeconds += Microseconds

		if type(event) == midi.SetTempoEvent:
			MicroSecondsPerQuarterNote = event.data[0]*256**2 + event.data[1]*256 + event.data[2]

		if (type(event) == midi.NoteOnEvent) and (event.channel == 0):
			if event.data[1] != 0: # press note
				note = event.data[0]
				A=np.vstack((A,[AbsoluteMicroSeconds,note]))
				# velocity = event.data[1]
			else: #release note
				note = event.data[0]+128
				A=np.vstack((A,[AbsoluteMicroSeconds,note]))

	C=1000000/24 # 24 notes per second (means 480/24 = 20)
	A[:,0]=(A[:,0]+C/2)/C
	A=A[1:,:] #remove first row
	A=A+1 #one based indexing now
	
	np.save("npynice/"+filename+".npy", A)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Converts a midifile to a [256,X] numpy array and stores it in npy/filename.npy")
	parser.add_argument('filename', action="store", help="as in mid/filename.mid")
	args = parser.parse_args()
	convert(args.filename)
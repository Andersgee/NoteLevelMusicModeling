import midi
import numpy as np

#filename = "bach_846_format0.mid"
#filename = "bach_847_format0.mid"
#filename="bach_850_format0.mid"
#filename="mid/BeethovenLudwigvan/appass_2_format0.mid"

#filename="appass_1_format0.mid"
filename="waldstein_1_format0.mid"
track = midi.read_midifile(filename)[0]

# default values
TicksPerQuarterNote = 480 # "Resolution"
MicroSecondsPerQuarterNote = 500000 # "Tempo" #120 quarternotes per minute is 1/(120/60/1000000)

#480*4/64 is the tick length of a 64th note
#480*4/32 is the tick length of a 32th note

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

#C=1000000/10 #tenth
#C=1000000/100 #centi
#C=1000000/1000 #milli
#C=1000000/20 # 20 notes per second (means 480/20 = 24)

C=1000000/24 # 24 notes per second (means 480/24 = 20)

A[:,0]=(A[:,0]+C/2)/C

print A[0:50,:]

print "shape", A.shape
print "last value", A[-1,0]
print "song should be", A[-1,0]/20, "seconds long"
np.save("test.npy",A)
import midi
import numpy as np
import argparse

def track2numpy(track):
    r = []
    acctick = 0
    for event in track:
        acctick = acctick + event.tick
        isoff = type(event) is midi.events.NoteOffEvent or (type(event) is midi.events.NoteOnEvent and event.data[1] == 0)
        if isoff:
            r.append([acctick, 128 + event.data[0]])
        if not isoff and type(event) is midi.events.NoteOnEvent:
            r.append([acctick, event.data[0]])
    return np.vstack(r)

def massage(filename):
	tracks = midi.read_midifile(filename)
	events = track2numpy(tracks[0]) # format0 files has everything in track0, probably easier to use standard format 1 and only a single instrument instead..
	events[:,0] = events[:,0] - events[0,0] #start tick counting on first note
	meantickspacing = float(events[-1,0]) / len(np.unique(events[:,0])) #many events can happen on same tick
	events[:,0] = (events[:,0] + meantickspacing/2) / meantickspacing #perhaps this is too harsh, I lose fine grained sequences this way..
	events+=1 #might aswell convert from 0-indexing to 1-indexing here
	return events

def convert(filename):
	#filename="elise_format0"
	a = massage("mid/"+filename+".mid")
	np.save("npy/"+filename+".npy", a)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Converts a midifile to a [256,X] numpy array and stores it in npy/filename.npy")
	parser.add_argument('filename', action="store", help="as in mid/filename.mid")
	args = parser.parse_args()
	convert(args.filename)


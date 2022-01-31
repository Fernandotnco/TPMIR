import mido
import numpy as np
import matplotlib.pyplot as plt
import string

from mido import MidiFile

def msg2dict(msg):
    result = dict()
    if 'note_on' in msg:
        on_ = True
    elif 'note_off' in msg:
        on_ = False
    else:
        on_ = None
    result['time'] = int(msg[msg.rfind('time'):].split(' ')[0].split('=')[1].translate(
        str.maketrans({a: None for a in string.punctuation})))

    if on_ is not None:
        for k in ['note', 'velocity']:
            result[k] = int(msg[msg.rfind(k):].split(' ')[0].split('=')[1].translate(
                str.maketrans({a: None for a in string.punctuation})))
    return [result, on_]

def switch_note(last_state, note, velocity, on_=True):
    # piano has 88 notes, corresponding to note id 21 to 108, any note out of this range will be ignored
    result = [0] * 88 if last_state is None else last_state.copy()
    if 21 <= note <= 108:
        if(not on_):
          result[note-21] = 0
        else:
          result[note-21] = 3 if velocity > 0 else 0
    return result

def get_new_state(new_msg, last_state):
    new_msg, on_ = msg2dict(str(new_msg))
    new_state = switch_note(last_state, note=new_msg['note'], velocity=new_msg['velocity'], on_=on_) if on_ is not None else last_state
    new_state [ np.array(new_state)].astype('uint32')
    return [new_state, new_msg['time']]
def track2seq(track):
    # piano has 88 notes, corresponding to note id 21 to 108, any note out of the id range will be ignored
    result = []
    last_state, last_time = get_new_state(str(track[0]), np.zeros((88,)).astype('uint32'))
    for i in range(1, len(track)):
        new_state, new_time = get_new_state(track[i], last_state)
        if new_time > 0:
            result.append(np.array(last_state))
            for i in range(new_time-1):
              result.append(np.array(last_state)%2)
        last_state, last_time = new_state, new_time
    return result

def mid2arry(mid, min_msg_pct=0.1):
    tracks_len = [len(tr) for tr in mid.tracks]
    min_n_msg = max(tracks_len) * min_msg_pct
    # convert each track to nested list
    all_arys = []
    for i in range(len(mid.tracks)):
        if len(mid.tracks[i]) > min_n_msg:
            ary_i = track2seq(mid.tracks[i])
            all_arys.append(ary_i)
    # make all nested list the same length
    max_len = max([len(ary) for ary in all_arys])
    for i in range(len(all_arys)):
        if len(all_arys[i]) < max_len:
            all_arys[i] += [[0] * 88] * (max_len - len(all_arys[i]))
    all_arys = np.array(all_arys)
    all_arys = all_arys.max(axis=0)
    # trim: remove consecutive 0s in the beginning and at the end
    sums = all_arys.sum(axis=1)
    ends = np.where(sums > 0)[0]
    return all_arys[min(ends): max(ends)]


def GetTempo(mid, MIDIArray):

    tempos = np.zeros((MIDIArray.shape[0],1)).astype('uint32')
    set_tempos = []
    for msg in mid.tracks[0]:
      if(msg.type == 'set_tempo'):
          set_tempos.append((msg.tempo, msg.time))

    curTime = 0
    for tempo in set_tempos:
        elapsed = tempo[1]
        if(elapsed > 0):
            tempos[curTime:curTime + elapsed] = int(curTempo)
        curTempo = tempo[0]
        curTime += elapsed
    if(tempos[-1] == 0):
        tempos[curTime:] = set_tempos[-1][0]
    return tempos


def CreatePianoRoll(mid, MIDIArray): 

    tempos = GetTempo(mid, MIDIArray)
    i = 0
    song = []
    while i < MIDIArray.shape[0]:

        mspb = (tempos[i] // 4000)[0] # miliseconds per sixteenth beat
        tol = mspb // 2


        curBeat = np.zeros((88,))
        for j in range(MIDIArray[i].shape[0]):
            curBeat[j] = np.max(MIDIArray[np.max([0, i-tol+1]): np.min([i+tol-1, MIDIArray.shape[0] -1]), j])

        song.append(curBeat)


        i += mspb
    song = np.array(song)

    for i in range(song.shape[0]):
        for j in range(song.shape[1]):
            if(song[i,j] == 3):
                song[i,j] = 1
            elif(song[i,j] == 1):
                song[i,j] = 0.5
            else:
                song[i,j] = 0


    return song

def SegmentSong(song):

    numSegs = song.shape[0] // 64
    segs = []
    numBeats = 0
    for i in range(numSegs):
        segs.append(song[i*64 : (i+1)*64, :])
        numBeats += 64
    if(numBeats < song.shape[0]):
        finalSeg = song[numBeats:, :]
        segs.append(np.concatenate((finalSeg, (np.zeros((64-(song.shape[0]-numBeats), 88))))))

    for i in range(len(segs)):
        segs[i] = segs[i].T
    return segs


def ProcessMIDI(fileName, plotMIDI = False):
    mid = MidiFile(fileName, clip=True)
    MIDIArray = mid2arry(mid)
    if(plotMIDI):
        plt.plot(range(MidiFile.shape[0]), np.multiply(np.where(MidiFile>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
        plt.title(fileName)
        plt.show()
    pianoRoll = CreatePianoRoll(mid, MIDIArray)
    segments = SegmentSong(pianoRoll)
    return segments

if __name__ == '__main__':
    segments = ProcessMIDI('ClassicPianoMIDI/tschai/ty_september.mid')
    plt.imshow(segments[0], cmap = 'gray')
    plt.show()


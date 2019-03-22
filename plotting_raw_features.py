import numpy as np
import pandas as pd
import scipy.io.wavfile
import scipy.signal
import matplotlib.pyplot as plt
import keras
from random import sample 
import librosa
from sklearn.preprocessing import normalize
from matplotlib import cm

# This file generates plots for the non-normalized features used in section 3.1.3

metadata = pd.read_csv('musicnet_metadata.csv')
time_info = []
index=0
for row in metadata.index:
    t = metadata.iloc[row].loc['seconds']
    for i in range(0,t,20):
	if (i+20>t):
	    break
        time_info.append(pd.DataFrame([[index,metadata.iloc[row].loc['composer'],metadata.iloc[row].loc['id'],i]], columns=['id','composer','song_id','time_offset']))
        index += 1
time_info = pd.concat(time_info)
time_info.index = time_info['id']
del time_info['id']

src_train = [int(i.split('.')[0]) for i in os.listdir('musicnet/train_labels')]
src_test = [int(i.split('.')[0]) for i in os.listdir('musicnet/test_labels')]
data_id_to_song_id = {i:time_info.iloc[i].loc['song_id'] for i in time_info.index}

FS = 44100
WINDOW = FS*20
bands = 30
scales = range(1,100)

def process_sample(sample):
    stft = scipy.signal.stft(sample, FS, window='hamm', nfft=2048, nperseg=2001)[2][:233,:]
    proc_stft = np.transpose(np.abs(stft))
    melspec = librosa.feature.melspectrogram(sample, sr=FS, n_fft=2048, n_mels=bands)
    logspec = librosa.amplitude_to_db((melspec))
    logspec = logspec.T
    output = np.zeros((logspec.shape[0], proc_stft.shape[1]))
    output[:proc_stft.shape[0],:] = proc_stft #normalize(proc_stft) #top left
    output[:,-bands:] = logspec #normalize(logspec) #bottom right

    return output


ID = 200
song_id = data_id_to_song_id[ID]
if song_id in src_train:
    filename = 'musicnet/train_data/%d.wav' % song_id
    
else:
    filename = 'musicnet/test_data/%d.wav' % song_id

    [rate, data] = scipy.io.wavfile.read(filename)
    data = data[time_info.iloc[ID].loc['time_offset']:time_info.iloc[ID].loc['time_offset']+WINDOW]

    # Store sample

    tmp = process_sample(data)


fig, ax = plt.subplots()#figsize=(100,1000))
stft_data = tmp[:833,:-30].T
cax = ax.imshow(stft_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
ax.figure.colorbar(cax, ax=ax)
ax.set_title('STFT without Normalize');

fig, ax = plt.subplots(figsize=(80,1))
mfcc_data = tmp[:,-bands:].T[:,500:1000]
cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
ax.figure.colorbar(cax, ax=ax)
ax.set_title('Mel Spectrogram without Normalize');

fig, ax = plt.subplots(figsize=(8,8))
mfcc_data = tmp.T[:,:]
cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
ax.figure.colorbar(cax, ax=ax)
ax.set_title('STFT + Mel Spectrogram without Normalize');



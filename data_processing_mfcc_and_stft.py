import os
import numpy as np
import pandas as pd
import scipy.io.wavfile
import scipy.signal
import matplotlib.pyplot as plt
import librosa
import pywt
from sklearn.preprocessing import normalize

# ASSUMES that it is called in a directory containing the musicnet folder

metadata = pd.read_csv('musicnet_metadata.csv')

train = [int(f.split('.')[0]) for f in os.listdir('musicnet/train_labels')]
test = [int(f.split('.')[0]) for f in os.listdir('musicnet/test_labels')]

FS = 44100
WINDOW = FS*20
bands = 30
scales = range(1,100)

def process_sample(sample):
  [f,t,stft] = scipy.signal.stft(sample, FS, window='hamm', nfft=2048, nperseg=2001)
  stft = stft[f<=5000,:]
  proc_stft = np.transpose(np.abs(stft))
  melspec = librosa.feature.melspectrogram(sample, sr=FS, n_fft=2048, n_mels=bands)
  logspec = librosa.amplitude_to_db((melspec))
  logspec = logspec.T
  output = np.zeros((logspec.shape[0], proc_stft.shape[1]))
  output[:proc_stft.shape[0],:] = normalize(proc_stft) #top left
  output[:,-bands:] = normalize(logspec) #bottom right
                 
  return output 

index = 1

composer_map = {metadata.iloc[i].id : metadata.iloc[i].composer for i in metadata.index}
composer_id = dict(zip(set(composer_map.values()),range(0,len(metadata.composer.unique()))))

print (composer_id)

labels = {}

filelist = train
for id in filelist:
    print ('processing id', id)
    filename = 'musicnet/train_data/%d.wav' % id
    [rate, data] = scipy.io.wavfile.read(filename)
    data = data[:-(data.shape[0]%WINDOW)]
    samples = np.split(data, data.shape[0]//WINDOW)
    assert rate==FS
    for sample in samples:
        current = process_sample(sample)
        np.savetxt('processed/%d.csv' % index,current,delimiter=',')
        index = index + 1
        labels[index] = [composer_id[composer_map[id]], id]

filelist = test
for id in filelist:
    print ('processing id', id)
    filename = 'musicnet/test_data/%d.wav' % id
    [rate, data] = scipy.io.wavfile.read(filename)
    data = data[:-(data.shape[0]%WINDOW)]
    samples = np.split(data, data.shape[0]//WINDOW)
    assert rate==FS
    for sample in samples:
        current = process_sample(sample)
        np.savetxt('processed/%d.csv' % index,current,delimiter=',')
        index = index + 1
        labels[index] = [composer_id[composer_map[id]], id]

pd.DataFrame(labels).to_csv('processed/labels.csv')

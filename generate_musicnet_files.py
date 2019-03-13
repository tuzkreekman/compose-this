import os
import numpy as np
import pandas as pd
import scipy.io.wavfile
import scipy.signal
import matplotlib.pyplot as plt

# ASSUMES that it is called in a directory containing the musicnet folder

metadata = pd.read_csv('musicnet/musicnet_metadata.csv')

train = [int(f.split('.')[0]) for f in os.listdir('musicnet/train_labels')]
test = [int(f.split('.')[0]) for f in os.listdir('musicnet/test_labels')]

FS = 44100
WINDOW = FS*20

process_sample = lambda sample: np.abs(scipy.signal.stft(sample, FS, window='hamm', nfft=2048, nperseg=2001)[2][:233,:])

index = 1

composer_map = {metadata.iloc[i].id : metadata.iloc[i].composer for i in metadata.index}
composer_id = dict(zip(set(composer_map.values()),range(0,len(metadata.composer.unique()))))

print composer_id

labels = {}

filelist = train
for id in filelist:
    print 'processing id', id
    filename = 'musicnet/train_data/%d.wav' % id
    [rate, data] = scipy.io.wavfile.read(filename)
    data = data[:-(data.shape[0]%WINDOW)]
    samples = np.split(data, data.shape[0]//WINDOW)
    assert rate==FS
    for sample in samples:
        current = process_sample(sample)
        np.savetxt('/tmp/zainabk/processed/%d.csv' % index,current,delimiter=',')
        index = index + 1
        labels[index] = composer_id[composer_map[id]]

filelist = test
for id in filelist:
    print 'processing id', id
    filename = 'musicnet/test_data/%d.wav' % id
    [rate, data] = scipy.io.wavfile.read(filename)
    data = data[:-(data.shape[0]%WINDOW)]
    samples = np.split(data, data.shape[0]//WINDOW)
    assert rate==FS
    for sample in samples:
        current = process_sample(sample)
        np.savetxt('/tmp/zainabk/processed/%d.csv' % index,current,delimiter=',')
        index = index + 1
        labels[index] = composer_id[composer_map[id]]

pd.Series(labels).to_csv('/tmp/zainabk/processed/labels.csv')




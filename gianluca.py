''' 
  Based on the Neural network architecture from: 

    Gianluca Micchi. A neural network for composer classification. 
    International Society for Music Information Retrieval Conference (ISMIR 2018), 2018, Paris, France. <hal-01879276>

  and Keras code from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
'''


import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import DataGenerator

DATASET_SIZE = 5978
TRAIN_SIZE = int(.7*DATASET_SIZE)

# Parameters
params = {'dim': (883,223),
          'batch_size': 64,
          'n_classes': 10,
          'n_channels': 1,
          'shuffle': True}

trainIDs = sample(range(DATASET_SIZE)+1, TRAIN_SIZE)
valIDS = [i+1 for i in range(DATASET_SIZE) if (i+1) not in trainIDs]

# Datasets
partition = {'train': trainIDs, 'validation': valIDs} # IDs
labels = np.genfromtxt('/tmp/zainabk/processed/labels.csv', delimiter=',') # Labels

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

# Design model
model = Sequential()

model.add(Conv2D(8, (16, 8), input_shape = (883, 223, 1)))
model.add(MaxPooling2D(pool_size=(4, 2)))
model.add(BatchNormalization())

model.add(Conv2D(8, (16, 8)))
model.add(MaxPooling2D(pool_size=(4, 2)))
model.add(BatchNormalization())

model.add(Conv2D(16, (16, 8)))
model.add(MaxPooling2D(pool_size=(4, 2)))
model.add(BatchNormalization())

model.add(LSTM(10))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)



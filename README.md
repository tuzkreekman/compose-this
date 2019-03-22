# compose-this

- generate_musicnet_files.py - create STFT feature set from Musicnet data

- nameThatComposer.ipynb - STFT feature set, CNN-LSTM, all composers, can also be used to do Beethoven vs rest, and more combos

- data_processing_mfcc_and_stft.ipynb - generates STFT + Mel Spectrum image feature set from Musicnet data

- piano-composer-detection.ipynb - uses parts of STFT + Mel Spectrum image features to train CNN-LSTM ("Piano Composer Problem")

- gianluca.py - has Keras implementation of network from [5]
   - Based on the Neural network architecture from: 
   
   ``` Gianluca Micchi. A neural network for composer classification. 
    International Society for Music Information Retrieval Conference (ISMIR 2018), 2018, Paris, France. <hal-01879276>
   ``` 
   and Keras code from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly


- data_generator.py - SOURCE: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly 

- plotting_normalized_features.py - Plots for 2.1.3 with final feature processing set

- plotting_raw_features.py - Plots for 2.1.3 with Mel + STFT but no normalization

- cnn-lstm-MFCInput.ipynb - Code for running cnn-lstm with an input of MFC and for the entire generalized composer dataset


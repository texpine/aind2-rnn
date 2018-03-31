import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras

# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    for start_pos in range(0, len(series)-window_size):
        X.append(series[start_pos:start_pos+window_size])
        y.append([series[start_pos+window_size]])

    return np.asarray(X), np.asarray(y)

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential([
        LSTM(5, input_shape=(window_size,1)),
        Dense(1)        
    ])
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
import re
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']    
    regex_ready = ''.join(['\\'+i if i in '[\^$.|?*+()\{\}' else i for i in punctuation])
    text = text.lower()    
    text = re.sub('[^a-z{}]+'.format(regex_ready), ' ', text)    
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    for start_pos in range(0, len(text)-window_size, step_size):
        inputs.append(text[start_pos:start_pos+window_size])
        outputs.append(text[start_pos+window_size])

    return inputs, outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
from keras.layers import Activation
def build_part2_RNN(window_size, num_chars):
    model = Sequential([
        LSTM(200, input_shape=(window_size, num_chars)),
        Dense(num_chars),
        Activation('softmax')
    ])
    return model

# from config import MODEL_NAME, NUMBER_OF_NODES, NUMBER_OF_MODELS, ONLY_FORMAT_DATA, TRAIN_DATA, TEST_DATA, WINDOW_SIZES, POST_TRAIN_PLOT, INPUT_COLUMNS, TARGET_COLUMNS
import pandas as pd
# from formatter2 import NUMBER_OF_NODES
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot
from keras.models import load_model
from keras import optimizers
import math
import argparse

RANGE = 25.00
LSTM_UNITS = 50
THRESHOLD = 600

INPUT_COLUMNS = []
for i in range(16):
    for j in range(8):
        INPUT_COLUMNS.append("p_" + str(i) + "_" + str(j))
TARGET_COLUMNS = ["letter"]
WINDOW_SIZES = [50]

MODEL_NAME = "LetterClassifier"
NUMBER_OF_MODELS = 1

POST_TRAIN_PLOT = True


# split the data sequence based the steps
def split_sequence(sequence, n_steps, inputColumns, targetColumns, offset = 0):
    
    X, Y = list(), list()
    n = len(sequence)
    for i in range(n):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > n-1:
            break
        # gather input'and output parts of the pattern
        seq_x, seq_y = sequence[inputColumns].iloc[i:end_ix], sequence[targetColumns].iloc[end_ix + offset]
        X.append(seq_x)
        Y.append(seq_y)
        if i%100==0:
            print("Sequencing:", str(round(float(i)/n*100,2)) + "% completed")
    return X, Y


# create input and target dataset
def create_data_set(df, window_size, inputCols, targetCols, offset=0):

    input_data, targets = split_sequence(df, window_size,inputCols,targetCols,offset)
    print('dataframe sequenced')

    return np.array(input_data), np.array(targets)


def split_train_test_data(df, model_name):
    '''
    given the pair wise meeting times of the nodes: meeting_times
    create the proper data set
    split the data into training and testing data set
    store the training and testing data set as npy files
    '''
    input_data, output_data = create_data_set(
        df, WINDOW_SIZE,INPUT_COLUMNS,TARGET_COLUMNS)

    # create training and testing data set
    n = len(input_data)
    x_train = input_data[:math.floor(n*0.8)]
    x_test = input_data[math.floor(n*0.8):]

    y_train = output_data[:math.floor(n*0.8)]
    y_test = output_data[math.floor(n*0.8):]

    np.save('x_' + model_name + '_train_' + str(WINDOW_SIZE)+ '.npy', x_train)
    np.save('x_' + model_name + '_test_' + str(WINDOW_SIZE)+ '.npy', x_test)
    np.save('y_' + model_name + '_train_' + str(WINDOW_SIZE)+ '.npy', y_train)
    np.save('y_' + model_name + '_test_' + str(WINDOW_SIZE)+ '.npy', y_test)
    return x_train, y_train

def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels

def keras_make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)

  return ds

def train(model_name):
    '''
    Build a LSTM and train with the training data set
    '''
    # try to read from the preprocessed numpy file
    try:
        x_train = np.load("./x_" +
                        model_name + "_train_" + str(WINDOW_SIZE)+ ".npy", allow_pickle=True)
        y_train = np.load("./y_" +
                        model_name + "_train_" + str(WINDOW_SIZE)+ ".npy", allow_pickle=True)

        x_test = np.load('./x_' + 
                        model_name + '_test_' + str(WINDOW_SIZE)+ '.npy', allow_pickle=True)

        y_test = np.load('./y_' + 
                        model_name + '_test_' + str(WINDOW_SIZE)+ '.npy', allow_pickle=True)
        print("Successfully read training data: ")
        # for i in range(5):
        #     print("x train, y_train:\n ", x_train[i], y_train[i])
    except:
        print("No relevant numpy files exist, parsing the original csv file now...")
        # parsing the dataframe to create the training and testing data set
        # dfCollection = []
        # for i in range(NUMBER_OF_NODES):
        #     df = pd.read_csv("../CSVs/" + model_name + "/" + model_name + "-node-" + str(i) + "-LSTMData-Normalized.csv", sep=",")
        #     dfCollection.append(df)

        df = pd.read_csv("letter.csv")

        columnNames = ['id', 'letter', 'next_id', 'word_id', 'position', 'fold']
        for i in range(16):
            for j in range(8):
                columnNames.append('p_' + str(i) + '_' + str(j))
        df.columns = columnNames
        df = df.head(10000)
        df.to_csv("./formatted_data.csv")

        x_train, y_train = split_train_test_data(
            df, model_name)

    input_features = len(INPUT_COLUMNS)
    output_features = len(TARGET_COLUMNS)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], input_features))

    for i in range(NUMBER_OF_MODELS):
        # ==== creating the simple LSTM model here
        model = keras.Sequential()
        model.add(tf.keras.layers.LSTM(LSTM_UNITS,
                                    activation='relu',
                                    input_shape=(WINDOW_SIZE, input_features)))
        # model.add(tf.keras.layers.Dropout(0.2))
        # this have to be modified based on the number of outputs we are trying to predict
        model.add(layers.Dense(units=output_features))
        optimizer = keras.optimizers.Adam(clipvalue=0.5)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        # model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
        es = EarlyStopping(monitor="val_loss", mode="min",  verbose=1, patience=5)

        # ==== train LSTM with x_train and y_train, validate with x_val, y_val
        model_history = model.fit(
            x_train, y_train,
            batch_size=64,
            epochs=100,
            verbose=1,
            validation_split=0.33,
            callbacks=[es])
        model.save(model_name + "-ws" + str(WINDOW_SIZE)+ "-" + str(i))
        print("The model has been trained for " + model_name + "-" + str(i))

        # ==== plot the training history
        print(model_history.history)
        if POST_TRAIN_PLOT:
            pyplot.plot(model_history.history['loss'], label='train')
            pyplot.plot(model_history.history['val_loss'], label='test')
            pyplot.legend()
            pyplot.show()

for i in range(len(WINDOW_SIZES)):
    WINDOW_SIZE = WINDOW_SIZES[i]
    print("Training", MODEL_NAME, "with a window size of", WINDOW_SIZE)
    train(MODEL_NAME)

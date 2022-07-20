'''
Convolutional neural network code for our handwriting classifier.
Author: Carl Gross
'''

import pandas as pd
import random
import tensorflow as tf
import time
import numpy as np

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


def split_data(df, train_proportion):
     ''' Inputs
             * df: dataframe containing data
             * train_proportion: proportion of data in df that will be used for
                 training. 1-train_proportion is proportion of data to be used
                 for testing
         Output
             * train_df: dataframe containing training data
             * test_df: dataframe containing testing data
     '''
     # Make sure there are row numbers
     df = df.reset_index(drop=True)

     # Reorder examples and split data according to train proportion
     train = df.sample(frac=train_proportion, axis=0)
     test = df.drop(index=train.index)
     return train, test

def format_data(filename):
    '''
    Adds column names as well as converts letter feature to indexes (0-25)
    '''
    df = pd.read_table(filename + '.data', delimiter="\t", header=None)

    columnNames = ['id', 'letter', 'next_id', 'word_id', 'position', 'fold']
    for i in range(16):
        for j in range(8):
            columnNames.append('p_' + str(i) + '_' + str(j))
    columnNames.append('Extra')
    df.columns = columnNames
    df.letter = [ ord(x)-ord("a") for x in df.letter ]

    df.to_csv(filename + '.csv',index=False)

def generate_model():
    '''
    Generates the convolutional neural network model with the given structure/layers.
    You can modify this function to change the structure of the neural network.
    '''
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(16, 8, 1)))
    model.add(layers.AveragePooling2D((2, 2)))
    # model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='sigmoid'))

    # model.summary()

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(26))

    # model.summary()
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return model

# We have formatted the data in order to isolate the features and target we hope to use
random.seed(time.time())

#establishes target column and features
target = ["letter"]
features = []
for i in range(16):
    for j in range(8):
        features.append("p_" + str(i) + "_" + str(j))

filename = 'letter'

# format_data(filename) #reformats data with headers and into csv form (only needs to be run once)

df = pd.read_csv(filename + '.csv')

#splitting the data into testing and training
train_proportion = 0.8
train_df, test_df = split_data(df, train_proportion)

train_images, train_labels = train_df[features], train_df[target]
test_images, test_labels = test_df[features], test_df[target]

#reshaping the data from a linear array into a 2D array of pixel mappings
train_images = np.array(train_images).reshape(len(train_images), 16, 8)
test_images = np.array(test_images).reshape(len(test_images), 16, 8)

iterations = 100

cnn = generate_model()

#terminates the fitting if loss doesnt improve for 4 epochs
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)

history = cnn.fit(train_images, train_labels, epochs=iterations, validation_data=(test_images, test_labels), callbacks=[callback])

test_loss, test_acc = cnn.evaluate(test_images,  test_labels, verbose=2)
print("\n###########################\n")
print("Test Loss:", test_loss, "; Test Accuracy:", test_acc)

#Training Accuracy Plot
plt.plot(history.history['accuracy'], label='training')
plt.plot(history.history['val_accuracy'], label = 'validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.savefig('plots/accuracyPlot.png')

plt.clf()

#Training Loss Plot
plt.plot(history.history['loss'], label='training')
plt.plot(history.history['val_loss'], label = 'validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.savefig('plots/lossPlot.png')






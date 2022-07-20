'''
Wesleyan University, COMP 343, Spring 2022
Tensorflow basic example
'''


import pandas as pd
import random
import tensorflow as tf
import time


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


# Load data and normalize
random.seed(time.time())
target = "letter"
features = []
for i in range(16):
    for j in range(8):
        features.append("p_" + str(i) + "_" + str(j))

filename = 'letter.csv'
df = pd.read_csv(filename)

columnNames = ['id', 'letter', 'next_id', 'word_id', 'position', 'fold']
for i in range(16):
    for j in range(8):
        columnNames.append('p_' + str(i) + '_' + str(j))
df.columns = columnNames

df.to_csv("./formatted_data.csv")

num_examples = df.shape[0]
num_features = df.shape[1]

# Set values to be either 0 and 1 by changing -1 values
# df.loc[df.Survived == -1] = 0

# Normalize features by max-min difference
# f = df[features]
# f = (f - f.min()) / (f.max() - f.min())
# df.update(f)
# print(df)


# Split data into training and test
train_proportion = 0.7
train_df, test_df = split_data(df, train_proportion)

num_folds = 5
num_hidden = 6
num_epochs = 10
learning_rate = 0.01


# normalize features
# one hidden layer with num_hidden nodes, tanh activation
# one output layer with 1 node for binary output, sigmoid activation
model = tf.keras.Sequential([
    tf.keras.layers.Normalization(axis=-1),
    tf.keras.layers.Dense(num_hidden, kernel_regularizer=None, activation='tanh'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'])


# Use Survived column for Y, other columns for X
X_train = train_df[features].to_numpy()
Y_train = train_df[target].to_numpy()
X_test = test_df[features].to_numpy()
Y_test = test_df[target].to_numpy()

history = model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test))
# predictions = model.predict(X_test)
# pred = np.array([1 if x >= .5 else 0 for x in predictions.flatten()]).reshape(Y_test.shape)
# print(pred, Y_test, (len(pred)-np.sum(np.abs(pred-Y_test)))/len(pred))
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)
print('Final Test Accuracy:', test_acc)



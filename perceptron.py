# Decided to switch to a multilayer perceptron instead of a convolutional neural network or an LSTM
# Author: Carl

import numpy as np
import pandas as pd

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.metrics import roc_auc_score, accuracy_score
# s = tf.InteractiveSession()

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

def to_numpy(df):
    a = df.to_numpy()
    return a.T

def get_X_y_data(df, features, target):
    ''' Split dataframe into X and y numpy arrays '''
    X_df = df[features]
    Y_df = df[target]
    X = to_numpy(X_df)
    Y = to_numpy(Y_df)
    return X, Y

def plot(epochs, training_accuracy, testing_accuracy):
    iterations = list(range(epochs))
    plt.plot(iterations, training_accuracy, label='Train')
    plt.plot(iterations, testing_accuracy, label='Test')
    plt.ylabel('Accuracy')
    plt.xlabel('iterations')
    plt.show()
    print("Train Accuracy: {0:.2f}".format(training_accuracy[-1]))
    print("Test Accuracy:{0:.2f}".format(testing_accuracy[-1]))

def train(X_train, y_train, X_test, y_test):
    s = tf.InteractiveSession()
    num_classes = y_train.shape[1]
    num_features = X_train.shape[1]
    num_output = y_train.shape[1]
    num_layers_0 = 512
    num_layers_1 = 256
    starter_learning_rate = 0.001
    regularizer_rate = 0.1

    # Placeholders for the input data
    input_X = tf.placeholder('float32',shape =(None,num_features),name="input_X")
    input_y = tf.placeholder('float32',shape = (None,num_classes),name='input_Y')
    ## for dropout layer
    keep_prob = tf.placeholder(tf.float32)

    # Weights initialized by random normal function with std_dev = 1/sqrt(number of input features)
    weights_0 = tf.Variable(tf.random_normal([num_features,num_layers_0], stddev=(1/tf.sqrt(float(num_features)))))
    bias_0 = tf.Variable(tf.random_normal([num_layers_0]))
    weights_1 = tf.Variable(tf.random_normal([num_layers_0,num_layers_1], stddev=(1/tf.sqrt(float(num_layers_0)))))
    bias_1 = tf.Variable(tf.random_normal([num_layers_1]))
    weights_2 = tf.Variable(tf.random_normal([num_layers_1,num_output], stddev=(1/tf.sqrt(float(num_layers_1)))))
    bias_2 = tf.Variable(tf.random_normal([num_output]))

    # Calculating outputs using weights and biases
    hidden_output_0 = tf.nn.relu(tf.matmul(input_X,weights_0)+bias_0)
    hidden_output_0_0 = tf.nn.dropout(hidden_output_0, keep_prob)
    hidden_output_1 = tf.nn.relu(tf.matmul(hidden_output_0_0,weights_1)+bias_1)
    hidden_output_1_1 = tf.nn.dropout(hidden_output_1, keep_prob)
    predicted_y = tf.sigmoid(tf.matmul(hidden_output_1_1,weights_2) + bias_2)

    # Loss Function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicted_y,labels=input_y)) \
            + regularizer_rate*(tf.reduce_sum(tf.square(bias_0)) + tf.reduce_sum(tf.square(bias_1)))

    # Variable learning rate
    learning_rate = tf.train.exponential_decay(starter_learning_rate, 0, 5, 0.85, staircase=True)

    # Adam optimzer for finding the right weight
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,var_list=[weights_0,weights_1,weights_2, bias_0,bias_1,bias_2])

    # Accuracy Metric
    correct_prediction = tf.equal(tf.argmax(y_train,1), tf.argmax(predicted_y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ## Training parameters
    batch_size = 128
    epochs=14
    dropout_prob = 0.6
    training_accuracy = []
    training_loss = []
    testing_accuracy = []
    s.run(tf.global_variables_initializer())
    for epoch in range(epochs):    
        arr = np.arange(X_train.shape[0])
        np.random.shuffle(arr)
        for index in range(0,X_train.shape[0],batch_size):
            s.run(optimizer, {input_X: X_train[arr[index:index+batch_size]],
                            input_y: y_train[arr[index:index+batch_size]],
                            keep_prob:dropout_prob})
        training_accuracy.append(s.run(accuracy, feed_dict= {input_X:X_train, input_y: y_train,keep_prob:1}))
        training_loss.append(s.run(loss, {input_X: X_train, input_y: y_train,keep_prob:1}))
        
        ## Evaluation of model
        testing_accuracy.append(accuracy_score(y_test.argmax(1),s.run(predicted_y, {input_X: X_test,keep_prob:1}).argmax(1)))
        print("Epoch:{0}, Train loss: {1:.2f} Train acc: {2:.3f}, Test acc:{3:.3f}".format(epoch,training_loss[epoch],training_accuracy[epoch],testing_accuracy[epoch]))
    plot(epochs, training_accuracy, testing_accuracy)

target = ["letter"]
features = []
for i in range(16):
    for j in range(8):
        features.append("p_" + str(i) + "_" + str(j))

# Formatting Data:
# filename = 'letter.csv'
# df = pd.read_csv(filename)

# columnNames = ['index','id', 'letter', 'next_id', 'word_id', 'position', 'fold']
# for i in range(16):
#     for j in range(8):
#         columnNames.append('p_' + str(i) + '_' + str(j))

# df.columns = columnNames

# df.to_csv("./letter.csv")

# Splitting Data
df = pd.read_csv('letter.csv')

df_train, df_test = split_data(df,0.8)

X_train, y_train = get_X_y_data(df_train, features, target)
X_test, y_test = get_X_y_data(df_test, features, target)

train(X_train, y_train, X_test, y_test)
"""
deep_learning_model.py
This module provides functions for creating, training, and testing deep learning models.
The deep learning models constructed in this module consist of Convolutional Neural Networks (CNNs) and hybrid
CNN with Long Short-Term Memory (LSTM) networks. The models are used for sequence classification tasks,
particularly for detecting defects in software systems.

- Convolutional Neural Networks (CNNs) are not used in the current version of the project because the CNN-LSTM model
  achieved better results

Functions:
- create_CNN_model(input_shape): Creates a Convolutional Neural Network model with one-dimensional convolution layers.
                                (this model are not used in the current version of the project because the CNN-LSTM
                                model achieved better results).
- create_CNN_LSTM_model(vocab_size,embedding_dim, max_length, conv_filters=20, kernel_size=10, pool_size=2,
                        lstm_units=32): Creates a hybrid model that combines a Convolutional Neural Network (CNN) with
                                        a Bidirectional Long Short-Term Memory (LSTM) layer for sequence classification
                                        tasks.
- create_and_train_CNN_LSTM_model(features_X, labeled_y): Creates and trains a CNN-LSTM model.
- create_and_train_model(training_data_path): Loads the data from the specified project, trains the hybrid model and
                                            prints out the training accuracy and other metrics.
- testing_model(project_csv_path, model): Loads the data from the specified project, tests the trained model on this
                                        data and returns the evaluation metrics.

Authors: Ahmed Kittaneh
Date: 23/7/2023
"""

# import necessary libraries
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D,Embedding,LSTM,Bidirectional
import keras
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from utils.data_preprocessing import load_or_prepare_data


# Constants
# the metrics to be used for evaluation
EVALUATION_METRICS = ['tp', 'fp', 'tn', 'fn', 'accuracy', 'precision', 'recall', 'auc']

# Create a convolutional neural network model with one-dimensional convolution layers
def create_CNN_model(input_shape):
    """
        Creates a Convolutional Neural Network model with one-dimensional convolution layers,
        a max pooling layer, and a dense output layer.

        :param input_shape: A tuple representing the shape of the input data.

        :returns:
            A compiled keras model object.
        """
    # Initialize model object
    model = Sequential()

    # Add layers to the model
    model.add(Conv1D(filters=64, kernel_size=7, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=10))
    model.add(Conv1D(64, 7, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    # print model summary
    print(model.summary())
    return model

def create_CNN_LSTM_model(vocab_size,embedding_dim, max_length, conv_filters=20, kernel_size=10,
                          pool_size=2, lstm_units=32):
    """
    Creates a hybrid model that combines a Convolutional Neural Network (CNN) with a Bidirectional Long Short-Term
    Memory (LSTM) layer for sequence classification tasks. The input is first passed through an embedding layer to
    convert the integer-encoded input sequence into dense vectors. Then, a 1D CNN layer is added to extract features
    from the sequence, followed by a max pooling layer to reduce the dimensionality. Next, a Bidirectional LSTM layer
    is added to capture the context from both past and future. Finally, a Dense layer with a sigmoid activation
    function is added to produce the output prediction.

    :param vocab_size: The size of the vocabulary of the input data.
    :param embedding_dim: The dimension of the dense embedding vectors.
    :param max_length: The maximum length of the input sequences.
    :param conv_filters: The number of filters to use in the CNN layer.
    :param kernel_size: The size of the convolution window.
    :param pool_size: The size of the max pooling window.
    :param lstm_units: The number of LSTM units in the Bidirectional LSTM layer.

    :return: The compiled hybrid CNN-LSTM model.
    """
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
    model.add(Conv1D(filters=conv_filters, kernel_size=kernel_size, strides=1, padding='same', activation='tanh'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Bidirectional(LSTM(lstm_units, activation='tanh'), name='LSTM'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer="RMSprop", loss="binary_crossentropy",
                           metrics=[keras.metrics.TruePositives(name='tp'),
                                    keras.metrics.FalsePositives
                                    (name='fp'),
                                    keras.metrics.TrueNegatives(name='tn'),
                                    keras.metrics.FalseNegatives
                                    (name='fn'),
                                    keras.metrics.BinaryAccuracy
                                    (name='accuracy'),
                                    keras.metrics.Precision
                                    (name='precision'),
                                    keras.metrics.Recall(name='recall'),
                                    keras.metrics.AUC(name='auc')])
    return model

def create_and_train_CNN_LSTM_model(features_X, labeled_y):
    """
    Creates and train a CNN-LSTM model, which combines a Convolutional Neural Network (CNN) with a Bidirectional LSTM.

    :param features_X: the prepared data for training and testing the model which is a list of feature vectors of
                        180 columns for semantic features and 20 columns for traditional features for each JavaFile
                        instance row.
    :param labeled_y: labels for the data which is a list of binary labels indicating whether each instance is
                        defective (1) or clean (0).

    :return: Tuple containing the model object and the accuracy score of training data.
    """

    # Set parameters
    # vocab_size = 500
    vocab_size = 100000
    embedding_dim = 64
    # embedding_dim = 30
    max_length = 200
    # max_length = 20
    # conv_filters = 64
    conv_filters = 20
    kernel_size = 7
    # kernel_size = 10
    pool_size = 10
    # pool_size = 2
    # lstm_units = 24
    lstm_units = 32

    # batch_size = 32
    batch_size = 128
    epochs = 10

    # Create the model
    model = create_CNN_LSTM_model(vocab_size=vocab_size, embedding_dim=embedding_dim, max_length=max_length,
                                  conv_filters=conv_filters, kernel_size=kernel_size, pool_size=pool_size,
                                  lstm_units=lstm_units)

    # Train the model
    model.fit(features_X, labeled_y, batch_size=batch_size, epochs=epochs)

    accuracy = model.evaluate(features_X, labeled_y, verbose=0)

    return model, accuracy

def create_and_train_model(training_data_path):
    """
        This function loads the data from the specified project, trains the hybrid model that combines a Convolutional
        Neural Network (CNN) with a Bidirectional Long Short-Term Memory (LSTM) on it, and then prints out the training
        accuracy and other relevant metrics.

        :param training_data_path: The path to the CSV file for the training project. It assumes the project's name
                                    is the second part of the path when split by '/'.

        :return: model, accuracy:
                  model: The trained CNN-LSTM model.
                  accuracy: The training accuracy and other evaluation metrics of the model.
    """
    # load data from the training project
    features_X, labeled_y = load_or_prepare_data(training_data_path)

    # train the model
    model, accuracy = create_and_train_CNN_LSTM_model(features_X, labeled_y)
    print(f'Accuracy of training {training_data_path.split("/")[1]}:')
    # print accuracy of training
    for name, value in zip(model.metrics_names, accuracy):
        if name in EVALUATION_METRICS:
            if name == 'precision':
                precision = value
            elif name == 'recall':
                recall = value
            print("\t{}: {}".format(name, value), end="")
    try:
        f_1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f_1 = 0
    print(f'\tF1: {f_1}')

    return model

def testing_model(project_csv_path, model):
    """
        This function loads the data from the specified project to test the trained model on this data.
        It prints out the evaluation accuracy along with other metrics and returns these metrics in a list.

        :param project_csv_path: The path to the CSV file for the testing project. It assumes the project's name is the
                                 second part of the path when split by '/'.
        :param model: The trained model which will be used for testing.

        :return: A list containing the evaluation metrics in the order: accuracy, precision, recall, F1 score, and AUC.
    """

    # load data from the testing project
    features_X, labeled_y = load_or_prepare_data(project_csv_path)

    # test the model
    accuracy_metrics = model.evaluate(features_X, labeled_y, verbose=0)

    # print accuracy of testing and save accuracy, precision, recall, f1, auc to return them
    print(f'Accuracy of evaluation {project_csv_path.split("/")[1]}:')
    for name, value in zip(model.metrics_names, accuracy_metrics):
        if name in EVALUATION_METRICS:
            if name == 'accuracy':
                accuracy = value
            elif name == 'auc':
                auc = value
            elif name == 'precision':
                precision = value
            elif name == 'recall':
                recall = value
            print("\t{}: {}".format(name, value), end="")
    try:
        f_1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f_1 = 0.5
    print(f'\tF1: {f_1}')
    return [accuracy, precision, recall, f_1, auc]


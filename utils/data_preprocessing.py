"""
data_preprocessing.py
This module provides functions for preparing data for deep learning models.
Functions include extracting features and labels from data (JavaFile class instances), padding features to a maximum
length, balancing class distributions, preparing and preprocessing data for input into a deep learning model, and
load data if it is already preprocessed or prepared data if it is not preprocessed and saved before.

Functions:
- extract_features_and_label(data): Extracts features and labels from list of '~utils.ast_tokenization.JavaFile'.
- pad_features(features, max_length): Pads features to a maximum length and converts them to a 2D list.
- balance_classes(features_X, labeled_y): Balances class distributions using random oversampling for minority class.
- prepare_data(data): Converts data into a format that is suitable for input into Deep learning model.
- load_or_prepare_data(project_csv_path): Loads data if it is already preprocessed or prepared data if it is not
                                            preprocessed and saved before.
Authors: Ahmed Kittaneh
Date: 23/7/2023
"""

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from imblearn.over_sampling import RandomOverSampler

from utils.ast_tokenization import read_java_files_and_extract_ast_tokens
from utils.files_utils import *

# Extracts features and labels from list of '~utils.ast_tokenization.JavaFile'.
def extract_features_and_label(data):
    """
    This function select necessary data for DL-model from a list of JavaFile instances by
    returning the binary label, the sequence of tokens for each instance, and the traditional features.

    :param data: a list of :py:class:'~utils.ast_tokenization.JavaFile' instances.

    :return:
        bug_count: a list of binary labels indicating whether each instance is defective (1) or clean (0).
        sequence: a list of sequences of tokens, where each sequence corresponds to the tokens of a JavaFile instance.
        traditional_features: a list of traditional features for each instance.
    """
    bug_count = [d.bug_count for d in data]
    sequence = [d.sequence for d in data]
    # make the label binary (0 for clean, 1 for defective) and ignoring the number of bugs.
    bug_count = [0 if b == 0 else 1 for b in bug_count]
    traditional_features = [d.traditional_features for d in data]
    return bug_count, sequence, traditional_features

# Pads features to a maximum length and converts them to a 2D list.
def pad_features(features, max_length):
    """Pads features to make them of the same length by adding zeroes
    for length less than the identified length and removing extra length
    bigger than the identified length.

    :param features: (list): a list of feature vectors, where each feature vector is a list of numerical values.
    :param max_length: (int): the maximum length of the feature vectors after padding.

    :returns:
        padded_features (list): List of padded feature vectors.
    """
    # Find the maximum length of the sub-lists (token)
    max_len = 0
    for i in range(len(features)):
        for j in range(len(features[i])):
            if len(features[i][j]) > max_len:
                max_len = len(features[i][j])

    # make all sub-lists (token) of the same length (max_len) by adding zeroes
    # and also convert the list to 2D list (features_modified)
    features_modified = [[] for i in range(len(features))]
    for i in range(len(features)):
        for j in range(len(features[i])):
            if len(features[i][j]) < max_len:
                features_modified[i] = features_modified[i] + features[i][j] + [0] * (max_len - len(features[i][j]))
            else:
                features_modified[i] = features_modified[i] + features[i][j]

    # Pad the feature vectors to make them of the same length.
    padded_features = []
    for feature in features_modified:
        if len(feature) < max_length:
            # Pad the feature vector with zeroes.
            padding = [0] * (max_length - len(feature))
            padded_feature = feature + padding
        elif len(feature) > max_length:
            # Truncate the feature vector to the maximum length.
            padded_feature = feature[:max_length]
        else:
            # The feature vector is already of the maximum length.
            padded_feature = feature
        padded_features.append(padded_feature)
    return padded_features

# Class imbalance problem
def balance_classes(features_X, labeled_y):
    """
    Balance the class distribution in a labeled dataset by oversampling the minority class.
    to solve the class imbalance problem.
    using the RandomOverSampler from the imbalanced-learn package.

    :param features_X: a list of feature vectors.
    :param labeled_y: a list of labels corresponding to the feature vectors.

    :return:
        X_resampled: a numpy array of the resampled feature vectors.
        y_resampled: a numpy array of the resampled labels.
    """

    X = np.array(features_X)
    y = np.array(labeled_y)

    # count the number of samples in each class
    class_counts = dict()
    for c in set(y):
        class_counts[c] = sum(y == c)

    # calculate the desired number of samples for each class
    max_class_count = max(class_counts.values())
    desired_class_counts = {c: max_class_count for c in class_counts}

    # create an instance of RandomOverSampler
    ros = RandomOverSampler(sampling_strategy=desired_class_counts, random_state=42)

    # fit and transform the dataset
    X_resampled, y_resampled = ros.fit_resample(X, y)
    return X_resampled, y_resampled

# Convert data into a format that is suitable for a DL-model input.
def prepare_data(data):
    """
    This function prepares the data for input into a Deep Learning model
    by performing the following steps:
        1.Extracts the necessary data for the DL-model using the extract_features_and_label function.
        2.Pads the token sequences of semantic features to ensure they are of the same length using the pad_features
            function.
        3.Combinations of the traditional features and the padded token sequences of semantic features
        4.Balances the classes by oversampling the minority class using the balance_classes function.

    :param data: a list of :py:class:'~utils.ast_tokenization.JavaFile' instances.

    :return: a tuple of four arrays (features_X, labeled_y), where:
             - features_X: the prepared data for training and testing the model which is a list of feature vectors of
                            180 columns for semantic features and 20 columns for traditional features for each JavaFile
                             instance row.
             - labeled_y: labels for the data which is a list of binary labels indicating whether each instance is
                            defective (1) or clean (0).

    """

    # Extract the binary label and token sequence for each JavaFile instance in the data
    defective_label, sequence, traditional_features = extract_features_and_label(data)

    # Pad the token sequences to a fixed length and balance the classes
    padded_sequence = pad_features(sequence, 180)

    # Convert the traditional features from string to float
    traditional_features_float = [[float(j) for j in i] for i in traditional_features]

    # Combine the traditional features with the padded sequence features
    combined_features = np.concatenate((traditional_features_float, padded_sequence), axis=1)

    # Balance the classes by oversampling the minority class
    features_X, labeled_y = balance_classes(combined_features, defective_label)

    # Return the preprocessed data for training and testing the model
    return features_X, labeled_y

# load the prepared data if exists, otherwise prepare the data
# Used when training and testing the model
def load_or_prepare_data(project_csv_path):
    """
        This function loads or prepares the data necessary for training and testing the model. If the data has already
        been prepared for the specified project, it loads the data directly from storage. Otherwise, it proceeds to
        prepare the data by reading Java files, extracting and pre-processing for semantic and traditional features,
        and saving the results for future use.

        :param project_csv_path: The path to the CSV file for the project. It assumes the project's name is the second
                                part of the path when split by '/'.

        :return: a tuple of four arrays (features_X, labeled_y), where:
             - features_X: the prepared data for training and testing the model which is a list of feature vectors of
                            180 columns for semantic features and 20 columns for traditional features for each JavaFile
                             instance row.
             - labeled_y: labels for the data which is a list of binary labels indicating whether each instance is
                            defective (1) or clean (0).
        """

    # get project name
    project_name = project_csv_path.split("/")[1]
    # read prepared data is exists
    if is_data_prepared(project_name):
        features_X, labeled_y = load_prepared_data(project_name)
    else:
        # read java files and extract the semantic features and the traditional features then save them as a list of
        # instance of :py:class:'~utils.ast_tokenization.JavaFile'
        files = read_java_files_and_extract_ast_tokens(project_name, project_csv_path)

        # prepare data for training and testing the model
        features_X, labeled_y = prepare_data(files)

        # save prepared data
        save_prepared_data(features_X, labeled_y, project_name)

    return features_X, labeled_y

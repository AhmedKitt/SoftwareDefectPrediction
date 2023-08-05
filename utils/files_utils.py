"""
files_utils.py

This module contains utility functions for searching, reading from, and writing to files.

Functions:
- search_file: Search for a file with the given name in the specified directory
- write_to_csv: Write data to a CSV file
- save_prepared_data: Save prepared data to a specified directory
- load_prepared_data: Load prepared data from a specified directory
- is_data_prepared: Check if prepared data exists in a specified directory
- create_log_file: Create a log file for logging the search results

Authors: Ahmed Kittaneh
Date: 23/7/2023
"""

# Imports
import os
import logging
import numpy as np
import csv

# Constants
FILE_NOT_FOUND = -1
PREPARED_DATA_DIR = 'prepared_data'
DATASET_DIR = 'dataset'

def search_file(file_name, search_path='/',search_file_log = logging.getLogger("search_file_log")):
    """
        Searches for a file with the given file name in the specified search path.

        :param file_name: (str): The name of the file to search for.
        :param search_path: (str): The directory path to start searching from. Default is root directory ('/').
        :param search_file_log: (Logger): The logger object to record the search progress and results.

        :returns:
            str: The absolute path of the file if found, else returns 'FILE_NOT_FOUND'.

        The function searches for the file in the given search path recursively using os.walk() function.
        If the file is found, its path is returned. Otherwise, the function returns 'FILE_NOT_FOUND'.
    """
    # read the file only when it is not in the directory because it is slow to search for it
    for root, dirs, files in os.walk('.'):
        if file_name in files:
            search_file_log.info(f"Found {file_name} in {root}")
            return os.path.join(root, file_name)
    return FILE_NOT_FOUND

def write_to_csv(data, filename):
    """
        Write the given data to a CSV file with the specified filename.

        :param data: (list): A list of objects that contain the path, bug count, AST tokens,
                                and sequence for each Java file.
        :param filename: (str): The name of the file to write the data to.

        :returns:
            None
    """
    # Open the CSV file with write access
    with open(filename, 'w', newline='') as f:
        # Create a CSV writer object
        writer = csv.writer(f)
        # Write the header row with column names
        writer.writerow(["path", "bugs", "tokens", "sequence"])
        # Iterate over the data and write each row to the CSV file
        for item in data:
            # Convert the list of AST tokens to a string separated by slashes
            token = ""
            for t in item.tokens:
                token = token + "/" + t
            # Convert the list of sequence numbers to a string separated by slashes
            seq = ""
            for s in item.sequence:
                seq = seq + "/" + str(s)
            # Write the row to the CSV file
            writer.writerow([item.path, item.bug_count, token, seq])

def save_prepared_data(features_X, labeled_y,project_name):
    """
        Saves the prepared data into numpy files in the split_data directory.

        :param features_X: (numpy array): The input features.
        :param labeled_y: (numpy array): The target labels.

        :returns:
            None
    """
    # Create the prepared_data directory if it doesn't exist
    prepared_data_dir = os.path.join(PREPARED_DATA_DIR,project_name,PREPARED_DATA_DIR)
    if not os.path.exists(prepared_data_dir):
        os.makedirs(prepared_data_dir)

    np.save(os.path.join(prepared_data_dir,"features_X.npy"), features_X)
    np.save(os.path.join(prepared_data_dir,"labeled_y.npy"), labeled_y)

# Load prepared data
def load_prepared_data(project_name):
    """
        Loads the prepared data from the split_data directory.

        :param project_name: (str): The name of the project to load the data for.

        :returns: The loaded data which is training and test data for input features and target labels.
    """

    prepared_data_dir = os.path.join(PREPARED_DATA_DIR,project_name,PREPARED_DATA_DIR)

    # Load the numpy arrays from the directory
    features_X = np.load(f'{prepared_data_dir}/features_X.npy')
    labeled_y = np.load(f'{prepared_data_dir}/labeled_y.npy')

    # Return the loaded data
    return features_X, labeled_y

# Check if data is prepared
def is_data_prepared(project_name):
    """
        This function checks if the prepared data exists by checking the existence of the numpy arrays saved in
        the DATASET_DIR directory.

        :param project_name: (str): The name of the project to check the data for.

        :return: A boolean value that indicates whether the data is prepared or not.
    """

    prepared_data_dir = os.path.join(PREPARED_DATA_DIR,project_name,PREPARED_DATA_DIR)
    return os.path.exists(f'{prepared_data_dir}/features_X.npy') and\
           os.path.exists(f'{prepared_data_dir}/labeled_y.npy')


# Create log file
def create_log_file(log_file_name):
    """
        Creates a logger for a log file with the specified name.

        :param log_file_name: (str): The name of the log file.

        :return: logger : The logger for the log file.
    """
    # Set up logging for the second file
    logger = logging.getLogger(log_file_name)
    logger.setLevel(logging.INFO)

    # Create the logging directory if it doesn't exist
    if not os.path.exists('logging'):
        os.makedirs('logging')

    # Configure the logging module
    log_filename = os.path.join('logging', log_file_name + '.log')

    # Set up the handler to write log messages to the file
    handler = logging.FileHandler(log_filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Return the logger object for writing log messages to the file
    return logger

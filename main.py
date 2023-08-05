"""
Software Defect Prediction Project

main.py

This Python script is the main executable for the software defect prediction project. It is tasked with training and
testing prediction models for several software projects. Each project has its data distributed over several files
which are listed as constant variables.

The script makes use of deep learning models defined in 'utils.deep_learning_model'. The model is individually trained
for each project and subsequently tested on all other projects, thereby covering both Within-Project Defect Prediction
(WPDP) and Cross-Project Defect Prediction (CPDP) scenarios.

The results of these evaluations are captured in terms of multiple metrics such as accuracy, precision, recall,
F1 score, and AUC. These are stored in a pandas DataFrame and subsequently written to a CSV file 'results.csv'.

Authors: Ahmed Kittaneh
Date: 23/7/2023
"""

# Imports
from utils.deep_learning_model import *
import pandas as pd

# Constants
ANT_PRJS = ["dataset/ant-1.3/ant-1.3.csv", "dataset/ant-1.4/ant-1.4.csv", "dataset/ant-1.5/ant-1.5.csv",
            "dataset/ant-1.6/ant-1.6.csv", "dataset/ant-1.7/ant-1.7.csv"]
CAMEL_PRJS = ["dataset/camel-1.0/camel-1.0.csv", "dataset/camel-1.2/camel-1.2.csv", "dataset/camel-1.4/camel-1.4.csv",
              "dataset/camel-1.6/camel-1.6.csv"]
JEDIT_PRJS = ["dataset/jedit-3.2/jedit-3.2.csv", "dataset/jedit-4.0/jedit-4.0.csv", "dataset/jedit-4.1/jedit-4.1.csv",
              "dataset/jedit-4.2/jedit-4.2.csv", "dataset/jedit-4.3/jedit-4.3.csv"]
LOG4J_PRJS = ["dataset/log4j-1.0/log4j-1.0.csv", "dataset/log4j-1.1/log4j-1.1.csv", "dataset/log4j-1.2/log4j-1.2.csv"]
LUCENE_PRJS = ["dataset/lucene-2.0/lucene-2.0.csv", "dataset/lucene-2.2/lucene-2.2.csv",
               "dataset/lucene-2.4/lucene-2.4.csv"]
POI_PRJS = ["dataset/poi-1.5/poi-1.5.csv", "dataset/poi-2.0/poi-2.0.csv", "dataset/poi-2.5/poi-2.5.csv",
            "dataset/poi-3.0/poi-3.0.csv"]
SYNAPSE_PRJS = ["dataset/synapse-1.0/synapse-1.0.csv", "dataset/synapse-1.1/synapse-1.1.csv",
                "dataset/synapse-1.2/synapse-1.2.csv"]
VELOCITY_PRJS = ["dataset/velocity-1.4/velocity-1.4.csv", "dataset/velocity-1.5/velocity-1.5.csv",
                 "dataset/velocity-1.6/velocity-1.6.csv"]
XALAN_PRJS = ["dataset/xalan-2.4/xalan-2.4.csv", "dataset/xalan-2.5/xalan-2.5.csv", "dataset/xalan-2.6/xalan-2.6.csv",
              "dataset/xalan-2.7/xalan-2.7.csv"]
XERCES_PRJS = ["dataset/xerces-1.1/xerces-1.1.csv", "dataset/xerces-1.2/xerces-1.2.csv",
               "dataset/xerces-1.3/xerces-1.3.csv", "dataset/xerces-1.4.4/xerces-1.4.4.csv"]
ALL_PROJECTS_PATHS = ANT_PRJS + CAMEL_PRJS + JEDIT_PRJS + LOG4J_PRJS + LUCENE_PRJS + POI_PRJS + SYNAPSE_PRJS + VELOCITY_PRJS \
                     + XALAN_PRJS + XERCES_PRJS
RESULTS_EVALUATION_METRICS = ["accuracy", "precision", "recall", "f1-score", "auc"]

# Main function:
def main():
    # Save the name of the projects as a list
    all_projects_names = [project.split("/")[1] for project in ALL_PROJECTS_PATHS]

    # Create multi-level column index for the results as dataframe
    columns = pd.MultiIndex.from_product([all_projects_names, RESULTS_EVALUATION_METRICS],
                                         names=['Project', 'Metrics'])
    # Create multi-level row index for the results as dataframe
    index = pd.MultiIndex.from_product([all_projects_names])
    # Create empty dataframe for the results
    results_df = pd.DataFrame(columns=columns, index=index)

    # Loop to create and train individual models for each project and test each model on all the other projects that are
    # Within-Project Defect Prediction (WPDP) and Cross-Project Defect Prediction (CPDP) and save the results
    for training_project in ALL_PROJECTS_PATHS:
        training_project_name = training_project.split("/")[1]
        print(f'Training {training_project_name}...')
        model = create_and_train_model(training_project)
        # loop over all the projects to test the model on them
        for testing_project in ALL_PROJECTS_PATHS:
            testing_project_name = testing_project.split("/")[1]
            results_metrics = testing_model(testing_project, model)
            results_df.loc[testing_project_name, (training_project_name, RESULTS_EVALUATION_METRICS )]\
                = results_metrics
    results_df.to_csv('results.csv')

if __name__ == "__main__":
    main()

"""
ast_tokenization.py
This file contains functions for preprocessing Java files for a bug prediction task. The JavaFile class represents
a Java file, and its attributes include the path to the file, the number of bugs in the file form PROMISE dataset,
the tokens and sequences extracted from the file, and the traditional features of the file form PROMISE dataset. The
file also contains functions to configure logging, extract specific AST tokens from a tree, extract tokens from a Java
file, and read Java files and extract AST tokens. These functions are used to process a CSV file containing paths to
Java files, its traditional features, and their associated bug counts, and to create a dataset for bug prediction.

Classes:
JavaFile: Represents a Java file, with attributes including its file path, number of bugs, and token sequence.

Functions:
- configure_logging_files(): Configures logging files to record search results and error messages.
- extract_specific_ast_tokens_from_tree(tree): Extracts specific tokens from an abstract syntax tree
    (AST) of a Java file.
- extract_tokens_from_file(path): Extracts tokens from a Java file located at a given file path.
- read_java_files_and_extract_ast_tokens(project_name, project_as_csv_file): Reads a CSV file containing information on
    Java files in a project, extracts AST tokens from each file, and returns a list of JavaFile objects
    representing the successfully processed files.

Authors: Ahmed Kittaneh
Date: 23/7/2023
"""

# Importing the libraries
import csv
from utils.files_utils import create_log_file, DATASET_DIR, write_to_csv, search_file, FILE_NOT_FOUND
import javalang
import os
from keras.preprocessing.text import Tokenizer

# Constants
JAVA_SYNTAX_ERROR = -1
INCLUDE_EMPTY_TOKENS = False
PATH_COLUMN_INDEX = 0
BUG_COUNT_COLUMN_INDEX = 21
SOURCE_CODE_DIR = 'source_code'

class JavaFile:
    def __init__(self, path, bug_count, tokens, sequence, traditional_features):
        """
        Constructor for JavaFile class.

        :param path: a string representing the path of the Java file.
        :param bug_count: an integer representing the number of bugs in the Java file.
        :param tokens: a list of strings representing the tokens in the Java file.
        :param sequence: a list of lists representing the tokenized and encoded sequence of the Java file.
        :param traditional_features: a list of floats representing the traditional features of the Java file from
                PROMISE dataset.
        """
        self.path = path
        self.bug_count = bug_count
        self.tokens = tokens
        self.sequence = sequence
        self.traditional_features = traditional_features

    # string representations of the JavaFile object.
    def __repr__(self):
        """
        Returns the canonical string representation of JavaFile object.

        :return: str, the string representation of the JavaFile object.
        """
        return f"JavaFile, path is ({self.path}), and has ({self.bug_count}) bugs"

    # string representations of the JavaFile object.
    def __str__(self):
        """
        Returns the informal string representation of JavaFile object.

        :return: str, the string representation of the JavaFile object.
        """
        return f"JavaFile, path is ({self.path}), and has ({self.bug_count}) bugs"

# Configure log files for various stages of processing.
def configure_logging_files():
    """
    Configure log files for various stages of processing.

    This function creates log files for different stages of processing to help with debugging and error tracking.
    The created log files include:
    - location_file_log: log file for storing the paths of Java files that already have correct path in dataset.
    - search_file_log: log file for storing the paths of Java files that get Java files paths from search results
                        for Java files. that because they do not have correct path in dataset.
    - not_found_log: log file for storing the java file name of Java files that were not found during the search process.
    - non_int_bug_count_files_log: log file for storing Java files names with non-integer bug counts.
    - java_syntax_error_files_log: log file for storing the paths of Java files with syntax errors.
    - no_tokens_files_log: log file for storing the paths of Java files with empty tokens.

    :param: None

    :return: A tuple of file handles for the created log files.
    """
    location_file_log = create_log_file("location_file_log")
    search_file_log = create_log_file("search_file_log")
    not_found_log = create_log_file("not_found_log")
    non_int_bug_count_files_log = create_log_file("non_int_bug_count_files_log")
    java_syntax_error_files_log = create_log_file("java_syntax_error_files_log")
    no_tokens_files_log = create_log_file("no_tokens_files_log")
    return location_file_log, search_file_log, not_found_log, non_int_bug_count_files_log, \
           java_syntax_error_files_log, no_tokens_files_log

# extracts the tokens from AST tree of a Java file.
def extract_specific_ast_tokens_from_tree(tree):
    """
    This function ast_tokens takes in an AST (Abstract Syntax Tree) and extracts the tokens of specific node types from
     it. The function returns a list of tokens extracted from the AST. The extracted tokens are prefixed with the node
     type for which the token is extracted.

    The function iterates through each path and node in the AST and checks for specific node types using isinstance.
    If the node is of a specific type, the corresponding token is extracted and appended to the res list.
    The extracted tokens are prefixed with the node type.

    The node types for which tokens are extracted include:
    ReferenceType, MethodInvocation, MethodDeclaration, TypeDeclaration, ClassDeclaration, EnumDeclaration,
    IfStatement, WhileStatement, DoStatement, ForStatement, AssertStatement, BreakStatement, ContinueStatement,
    ReturnStatement, ThrowStatement, SynchronizedStatement, and TryStatement.
    these node are used in CNN model.

    But in CNN-LSTM model we use all node types.
    the previous node types and the following node types:
    SwitchStatement, BlockStatement, SwitchStatementCase, ForControl, EnhancedForControl, TryResource, CatchClause,
    CatchClauseParameter, ClassCreator, SuperMethodInvocation, FormalParameter, PackageDeclaration,
    InterfaceDeclaration, ConstructorDeclaration, VariableDeclarator

    :param tree: the AST (Abstract Syntax Tree) of a Java code file.

    :return: a list of strings representing the tokens in the given AST. Each token is represented as a string that
    starts with a specific prefix that indicates the type of the token (e.g. ReferenceType_, MethodInvocation_, etc.)
    """

    res = []
    for path, node in tree:
        pattern = javalang.tree.ReferenceType
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('ReferenceType')
        pattern = javalang.tree.MethodInvocation
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('MethodInvocation')
        pattern = javalang.tree.MethodDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('MethodDeclaration')
        pattern = javalang.tree.TypeDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('TypeDeclaration')
        pattern = javalang.tree.ClassDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('ClassDeclaration')
        pattern = javalang.tree.EnumDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('EnumDeclaration')
        pattern = javalang.tree.IfStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('ifstatement')
        pattern = javalang.tree.WhileStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('whilestatement')
        pattern = javalang.tree.DoStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('dostatement')
        pattern = javalang.tree.ForStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('forstatement')
        pattern = javalang.tree.AssertStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('assertstatement')
        pattern = javalang.tree.BreakStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('breakstatement')
        pattern = javalang.tree.ContinueStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('continuestatement')
        pattern = javalang.tree.ReturnStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('returnstatement')
        pattern = javalang.tree.ThrowStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('throwstatement')
        pattern = javalang.tree.SynchronizedStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('synchronizedstatement')
        pattern = javalang.tree.TryStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('trystatement')

        # the following node types are used in CNN-LSTM model
        pattern = javalang.tree.SwitchStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('switchstatement')
        pattern = javalang.tree.BlockStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('blockstatement')
        pattern = javalang.tree.SwitchStatementCase
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('switchstatementcase')
        pattern = javalang.tree.ForControl
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('forcontrol')
        pattern = javalang.tree.EnhancedForControl
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('enhancedforcontrol')
        pattern = javalang.tree.TryResource
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('tryresource')
        pattern = javalang.tree.CatchClause
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('catchclause')
        pattern = javalang.tree.CatchClauseParameter
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('catchclauseparameter')
        pattern = javalang.tree.ClassCreator
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('classcreator')
        pattern = javalang.tree.SuperMethodInvocation
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('supermethodinvocation')
        pattern = javalang.tree.FormalParameter
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('formalparameter')
        pattern = javalang.tree.PackageDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('packagedeclaration')
        pattern = javalang.tree.InterfaceDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('interfacedeclaration')
        pattern = javalang.tree.ConstructorDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('constructordeclaration')
        pattern = javalang.tree.VariableDeclarator
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('variabledeclarator')
    return res

# extracts the tokens from a Java file.
def extract_tokens_from_file(path):
    """
    This function opens the file specified by the path parameter and reads its contents as a string.
    It then attempts to parse the source code using the javalang.parse.parse function, which returns
    an abstract syntax tree (AST) object. If the source code contains no syntax errors,
    the extract_specific_ast_tokens_from_tree function is called to extract specific tokens from the AST.
    If the source code contains syntax errors, the function returns the constant JAVA_SYNTAX_ERROR.

    :param path: the path to the file containing the source code to extract tokens from.

    :return: a list of tokens extracted from the file, or the string "JAVA_SYNTAX_ERROR" if the file
                contains syntax errors
    """
    # Read the file
    # with open(path, 'r') as file:
    #     source_code = file.read()
    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
        source_code = file.read()
    try:
        # Parse the source code
        ast = javalang.parse.parse(source_code)
        # Extract tokens from the AST
        tokens = extract_specific_ast_tokens_from_tree(ast)
        return tokens
    # If the source code contains syntax errors, return the constant JAVA_SYNTAX_ERROR
    except javalang.parser.JavaSyntaxError:
        return JAVA_SYNTAX_ERROR

# Read Java files from a CSV file and extract their AST tokens
def read_java_files_and_extract_ast_tokens(project_name, project_as_csv_file):
    """
    This is a function that reads a CSV file containing paths to Java files, its traditional features, and their
    corresponding bug counts.
    For each file in the CSV, it attempts to extract the abstract syntax tree (AST) of the Java code using the
    extract_tokens_from_file function, and then converts the AST to a sequence of tokens using the Tokenizer class.
    If the file is not found, has a non-integer bug count, has a syntax error, or has no tokens after parsing,
    it is skipped and added to a list of unresolved files. Finally, the function writes the list of resolved and
    unresolved files to separate CSV files and returns a list of JavaFile objects that contain the path, bug count,
    list of tokens, token sequence, and traditional features for each successfully parsed file.

    :param project_name: (str) the name of the project to be used in the name of directories and files of the project
    :param project_as_csv_file: (str) the path to the CSV file containing the paths to the Java files, its traditional
                                features, and their corresponding bug counts.

    :return: list of JavaFile objects - a list of JavaFile objects that contain the path, bug count,
                list of tokens, token sequence, and traditional features for each successfully parsed file
    """

    # Initialize empty lists for successful and unsuccessful files
    files = []
    unresolved_files = []

    # Create log files for different types of logging
    location_file_log, search_file_log, not_found_log, non_int_bug_count_files_log, java_syntax_error_files_log, no_tokens_files_log \
        = configure_logging_files()

    # Open the CSV file and iterate over its rows
    with open(project_as_csv_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:

            # Get the path and bug count for the Java file
            path = row[PATH_COLUMN_INDEX].replace(".", "/") + ".java"
            # the path is relative to the project directory
            path = os.path.join(DATASET_DIR,project_name, SOURCE_CODE_DIR, path)
            bug_count = row[BUG_COUNT_COLUMN_INDEX]
            traditional_features = row[PATH_COLUMN_INDEX+1:BUG_COUNT_COLUMN_INDEX]

            # Skip files with non-integer bug count and log them
            try:
                bug_count = int(bug_count)
            except ValueError:
                # print(f"Skipping {path} due to non-integer bug count: {bug_count}")
                # logging.info(f"Skipping {path} due to non-integer bug count: {bug_count}")
                non_int_bug_count_files_log.info(f"Skipping {path} due to non-integer bug count: {bug_count}")
                unresolved_files.append(JavaFile(path, bug_count, "non-integer bug count", [], traditional_features))
                continue

            # Skip If the file does not exist and log it as unresolved if not found
            if not os.path.exists(path):
                # read the file only when it is not in the directory because it is slow to search for it
                jave_file_name = path.split("/")[-1]
                # not_found_log.info(f"Skipping {jave_file_name} due to file not found")
                # unresolved_files.append(JavaFile(jave_file_name, bug_count, "non-existing files", []))
                # continue

                # path = search_file(jave_file_name, search_path='/', search_file_log=search_file_log)
                path = FILE_NOT_FOUND
                if path == FILE_NOT_FOUND:
                    not_found_log.info(f"Skipping {jave_file_name} due to file not found")
                    unresolved_files.append(JavaFile(jave_file_name, bug_count, "non-existing files", [], traditional_features))
                    continue
            else:
                location_file_log.info(f"Found {path} in {os.getcwd()}")

            # Extract tokens from the file and skip if there is a syntax error
            tokens = extract_tokens_from_file(path)
            if tokens == JAVA_SYNTAX_ERROR:
                java_syntax_error_files_log.info(f"Skipping {path} due to syntax error")
                unresolved_files.append(JavaFile(path, bug_count, "syntax error", [], traditional_features))
            # or empty tokens if specified by INCLUDE_EMPTY_TOKENS flag
            elif not INCLUDE_EMPTY_TOKENS and tokens == []:
                no_tokens_files_log.info(f"Skipping {path} due to empty tokens")
                unresolved_files.append(JavaFile(path, bug_count, "empty tokens", [], traditional_features))
            else:
                # Tokenize the tokens and append the JavaFile object to the list of successful files
                tokenizer = Tokenizer(num_words=None, lower=False)
                tokenizer.fit_on_texts(tokens)
                sequence = tokenizer.texts_to_sequences(tokens)
                files.append(JavaFile(path, bug_count, tokens, sequence, traditional_features))

    # Print information about the number of successful and unsuccessful files
    print(f'======================={project_name}=======================')
    print("reading java files has been done")
    print(f'Number of files has been read is {len(files)}')
    write_to_csv(files, os.path.join(DATASET_DIR,project_name, "successful_read.csv"))
    print(f'Number of files has been read but not resolved is {len(unresolved_files)}')
    write_to_csv(unresolved_files, os.path.join(DATASET_DIR,project_name, "unresolved_files.csv"))
    print('\n')

    # Return the list of JavaFile objects
    return files

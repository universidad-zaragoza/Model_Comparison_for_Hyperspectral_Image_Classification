#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module for hypesrspectral image classification with different models.

Available models are:

1 - `Deep Convolutional Neural Networks for Hyperspectral Image
     Classification`
2 - `Hyperspectral Image Classification with Convolutional Neural
     Networks`
3 - `LightGBM`
4 - `SVM with scikit learn SVC`
5 - `SVM with scikit learn LinearSVC`
6 - `RandomForestClassifier with scikit learn`

"""
from __future__ import division, absolute_import, print_function
import sys
import os
import errno
import scipy.io
import numpy as np
import math
import json
import timeit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser, RawDescriptionHelpFormatter

import keras
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import joblib

from hyperspectral_preprocessing import preprocess_image

# Management of arguments
# =============================================================================

# Absolute path of the images directory
DATA_PATH = os.environ["HYPERSPECTRAL_DATA_PATH"]

# Name of the images information file
IMAGES_FILE_NAME = "images.json"

# Images information file
IMAGES_FILE = os.path.join(DATA_PATH, IMAGES_FILE_NAME)

# Absolute path of the working directory
WORKING_DIR = os.getcwd()

# Relative path of the output directory
OUTPUT_DIR = WORKING_DIR

# Name of the output file
OUTPUT_FILE_NAME = "output.txt"

# Output file
OUTPUT_FILE = os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME)

# Load random indexes
LOAD_INDEXES = False
INDEXES_DIR = None

# Available models
MODELS = {1: ("Deep Convolutional Neural Networks for Hyperspectral Image"
              " Classification"),
          2: ("Hyperspectral Image Classification with Convolutional Neural"
              " Networks"),
          3: "Lightgbm",
          4: "SVM with scikit learn SVC",
          5: "SVM with scikit learn LinearSVC",
          6: "RandomForestClassifier with scikit learn"}

# Models variables
MODEL = 1
JOIN_MODELS = False
MODEL1 = 1
MODEL2 = 1
ALL_TOGETHER = False

# Load trained models
TRAINED_MODELS = False
TRAINED_MODELS_DIR = ""
TRAINED_MODELS_DIR2 = ""
TRAINED_MODELS_DIRS = []

# Default train epochs for CNN models
TRAIN_EPOCHS = 100

# Nuber of best features to use
FEATURES = 0

# Flag to activate lightgbm cross validation
CROSS_VALIDATION = False

# lightgbm cross validation parameters
N_ESTIMATORS_LIST = [100, 200, 400, 800]
#N_ESTIMATORS_LIST = [1000, 1200, 1400, 1800]
#N_ESTIMATORS_LIST = [1800, 1850, 1900, 1950, 2000]
#N_ESTIMATORS_LIST = [50, 100, 150, 200, 250,
#                     300, 350, 400, 450, 500,
#                     550, 600, 650, 700, 750,
#                     800, 850, 900, 950, 1000,
#                     1050, 1100, 1150, 1200, 1250,
#                     1300, 1350, 1400, 1450, 1500,
#                     1550, 1600, 1650, 1700, 1750,
#                     1800, 1850, 1900, 1950, 2000]
MIN_CHILD_SAMPLES_LIST = [20, 50, 100, 200, 300]

def parse_args():
    """Analyzes the received parameters and returns them organized.
    
    Takes the lis of string received at sys.argv and generates a
    namespace asigning them to objects.
    
    Returns
    -------
    out: namespace
        The namespace with the values of the received parameters asigned
        to objects.
    
    """
    # Generate the parameter analyzer
    parser = ArgumentParser(description = __doc__,
                            formatter_class = RawDescriptionHelpFormatter)
    
    # Add arguments
    parser.add_argument("--data_path",
                        help="Absolute path of the images.")
    parser.add_argument("--working_dir",
                        help="Absolute path of the working directory.")
    parser.add_argument("-d", "--output_dir",
                        help="Relative path of the output directory. "
                             + "It will be added to the working directory.")
    parser.add_argument("-o", "--output",
                        help="Name of the output file. ")
    parser.add_argument("-i", "--indexes_dir",
                        help="Absolute path of random and train indexes "
                             + "directory.")
    parser.add_argument("-c", "--cross_validation",
                        action="store_true",
                        help="Flag to activate lightgbm cross validation.")
    parser.add_argument("-e", "--train_epochs",
                        type=int,
                        help="Number of training epochs for CNN models.")
    parser.add_argument("-f", "--features",
                        type=int,
                        help="Nuber of best features to use. If `0` (default) "
                             + "it uses every feature.")
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument("-m", "--model",
                        type=int,
                        choices=MODELS.keys(),
                        help="Number corresponding to the selected model.")
    group1.add_argument("-M", "--models",
                        nargs=2,
                        type=int,
                        choices=MODELS.keys(),
                        help="Number of the two models to combine.")
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument("-T", "--trained_models_dirs",
                        nargs=2,
                        help="Absolute path of trained models directories.")
    group2.add_argument("-t", "--trained_models_dir",
                        help="Absolute path of trained models directory.")
    group2.add_argument("-A", "--all_together",
                        nargs=4,
                        help="Absolute path of trained models directories.")
    
    # Return the analized parameters
    return parser.parse_args()

def use_args(args):
    """Realizes required actions depending on the received parameters.
    
    Parameters
    ----------
    args: namespace
        The namespace with the values of the received parameters asigned
        to objects.
        
    """
    global DATA_PATH
    global WORKING_DIR
    global OUTPUT_DIR
    global OUTPUT_FILE_NAME
    global OUTPUT_FILE
    global LOAD_INDEXES
    global INDEXES_DIR
    global MODEL
    global JOIN_MODELS
    global MODEL1
    global MODEL2
    global ALL_TOGETHER
    global TRAINED_MODELS
    global TRAINED_MODELS_DIR
    global TRAINED_MODELS_DIR2
    global TRAINED_MODELS_DIRS
    global CROSS_VALIDATION
    global TRAIN_EPOCHS
    global FEATURES
    
    if args.data_path:
        # Change the default path of the images
        DATA_PATH = args.data_path
    
    if args.working_dir:
        # Change the default path of the working directory
        WORKING_DIR = args.working_dir
        OUTPUT_DIR = WORKING_DIR
        OUTPUT_FILE = os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME)
    
    if args.output_dir:
        # Change the default path of the output directory
        OUTPUT_DIR = os.path.join(WORKING_DIR, args.output_dir)
        OUTPUT_FILE = os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME)
    
    if args.output:
        # Change the default name of the output file
        OUTPUT_FILE_NAME = args.output
        OUTPUT_FILE = os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME)
    
    if args.indexes_dir:
        # Load random and train indexes from file
        LOAD_INDEXES = True
        INDEXES_DIR = args.indexes_dir
    
    if args.model:
        # Select model
        MODEL = args.model
    
    if args.models:
        
        if not args.trained_models_dirs:
            raise Exception("Arg. `-M --models` requires arg. "
                            + "`-T --trained_models_dirs`")
        
        # Models to combine
        JOIN_MODELS = True
        MODEL1 = args.models[0]
        MODEL2 = args.models[1]
    
    if args.trained_models_dir:
        # Load trained models from file
        TRAINED_MODELS = True
        TRAINED_MODELS_DIR = args.trained_models_dir
    
    if args.trained_models_dirs:
        # Load trained models from file
        TRAINED_MODELS = True
        TRAINED_MODELS_DIR = args.trained_models_dirs[0]
        TRAINED_MODELS_DIR2 = args.trained_models_dirs[1]
    
    if args.all_together:
        # The four models together
        ALL_TOGETHER = True
        TRAINED_MODELS_DIRS = args.all_together
    
    if args.cross_validation:
        # Activate cross_validation
        CROSS_VALIDATION = True
    
    if args.train_epochs:
        # Change the default number of train epochs
        TRAIN_EPOCHS = args.train_epochs
    
    if args.features:
        # Nuber of best features to use
        FEATURES = args.features

# Preprocess functions
# =============================================================================

def model_1_parameters(num_features, num_classes, image_info):
    """Prepare the model training parameters.
    
    Model from the paper `Deep Convolutional Neural Networks for
    Hyperspectral Image Classification`.
    
    """
    parameters = {}
    parameters['n1'] = num_features
    parameters['k1'] = int(math.floor(parameters['n1'] / 9))
    parameters['n2'] = parameters['n1'] - parameters['k1'] + 1
    if image_info['key'][:5] == "pavia":
        parameters['n3'] = 30
        parameters['k2'] = 3
    else:
        parameters['n3'] = 40
        parameters['k2'] = 5
    parameters['n4'] = 100
    parameters['n5'] = num_classes
    
    return parameters

def model_2_parameters(num_features, num_classes):
    """Prepare the model training parameters.
    
    Model from the paper `Hyperspectral Image Classification with
    Convolutional Neural Networks`.
    
    """
    parameters = {}
    parameters['num_features'] = num_features
    parameters['num_classes'] = num_classes
    
    return parameters

def model_3_parameters(num_features, num_classes, image_info):
    """Prepare the model training parameters.
    
    Lightgbm model.
    
    """
    parameters = {}
    parameters['num_features'] = num_features
    parameters['num_classes'] = num_classes
    parameters['n_estimators'] = image_info['n_estimators']
    min_child_samples = image_info['min_child_samples']
    parameters['min_child_samples'] = min_child_samples
    
    # Parameters message
    with open(OUTPUT_FILE, 'a') as f:
        f.write("min_child_samples: {}\n\n".format(min_child_samples))
    
    return parameters

def model_parameters(num_features, num_classes, image_info):
    """Prepare the model training parameters.
    
    Chooses the parameters function depending on the selected model.
    
    """
    if MODEL == 3:
        return model_3_parameters(num_features, num_classes, image_info)
    elif MODEL == 6:
        return model_3_parameters(num_features, num_classes, image_info)
    elif MODEL == 1:
        return model_1_parameters(num_features, num_classes, image_info)
    else:
        # For models 2, 4 and 5
        return model_2_parameters(num_features, num_classes)

# Model generation functions
# =============================================================================

def get_model_1(parameters):
    """Generates the model.
    
    Model from the paper `Deep Convolutional Neural Networks for
    Hyperspectral Image Classification`.
    
    See Section 3 `CNN-Based HSI Classification` in the paper for an
    explanation about the structure and parameters.
        Default range of `random_uniform` initializer: [-0.05, 0.05]
        Default learning rate of `sgd` optimizer: 0.5
    """
    # Parameters
    n1 = parameters['n1']
    k1 = parameters['k1']
    n2 = parameters['n2']
    n3 = parameters['n3']
    k2 = parameters['k2']
    n4 = parameters['n4']
    n5 = parameters['n5']
    NUM_FILTERS_C1 = 20
    
    # Sequential model
    model = keras.models.Sequential()
    
    # Add C1 layer
    # ------------
    # Input_shape (batch, rows, cols, channels) = (-, n1, 1, 1)
    # Output_shape (batch, rows, cols, channels) = (-, n2, 1, NUM_FILTERS_C1)
    model.add(keras.layers.Conv2D(filters=NUM_FILTERS_C1,
                                  kernel_size=(k1, 1),
                                  padding='valid',
                                  data_format="channels_last",
                                  activation='tanh',
                                  kernel_initializer='random_uniform',
                                  bias_initializer='random_uniform',
                                  input_shape=(n1,1,1)))
    
    # Add M2 layer
    # ------------
    # Input_shape (batch, rows, cols, channels) = (-, n2, 1, NUM_FILTERS_C1)
    # Output_shape (batch, rows, cols, channels) = (-, n3, 1, NUM_FILTERS_C1)
    model.add(keras.layers.MaxPooling2D(pool_size=(k2, 1),
                                        padding='same',
                                        data_format="channels_last"))
    
    # Flatten before dense layer
    model.add(keras.layers.Flatten())
    
    # Add F3 layer
    # ------------
    # Intput_shape (batch, rows, cols, channels) = (-, n3 x 1 x NUM_FILTERS_C1)
    # Output_shape (batch, dim) = (-, n4)
    model.add(keras.layers.Dense(units=n4,
                                 activation='tanh',
                                 kernel_initializer='random_uniform',
                                 bias_initializer='random_uniform'))
    
    # Add F4 layer
    # ------------
    # Intput_shape (batch, dim) = (1, n4)
    # Output_shape (batch, dim) = (1, n5)
    model.add(keras.layers.Dense(units=n5,
                                 activation='softmax',
                                 kernel_initializer='random_uniform',
                                 bias_initializer='random_uniform'))
    
    # Compile model
    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    
    # Print the model summary to output file
    #     To print to stdout: model.summary()
    with open(OUTPUT_FILE, 'a') as f:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # Return the model
    return model

def get_model_2(parameters):
    """Generates the model.
    
    Model from the paper `Hyperspectral Image Classification with
    Convolutional Neural Networks`.
    
    """
    # Parameters
    BANDS = parameters['num_features']
    CLASSES = parameters['num_classes']
    
    # Sequential model
    model = keras.models.Sequential()
    
    # Add convolution (1)
    # -------------------
    # Input_shape (batch, rows, cols, channels) = (-, 9, BANDS, 1)
    # Output_shape (batch, rows, cols, channels) = (-, BANDS - 15, 1, 32)
    model.add(keras.layers.Conv2D(filters=32,
                                  kernel_size=(9, 16),
                                  padding='same',
                                  data_format="channels_last",
                                  activation='tanh',
                                  input_shape=(9, BANDS,1)))
    
    # Add convolution (2)
    # -------------------
    # Input_shape (batch, rows, cols, channels) = (-, BANDS - 15, 1, 32)
    # Output_shape (batch, rows, cols, channels) = (-, BANDS - 30, 1, 32)
    model.add(keras.layers.Conv2D(filters=32,
                                  kernel_size=(1, 16),
                                  padding='same',
                                  data_format="channels_last",
                                  activation='tanh'))
    
    # Add convolution (3)
    # -------------------
    # Input_shape (batch, rows, cols, channels) = (-, BANDS - 30, 1, 32)
    # Output_shape (batch, rows, cols, channels) = (-, BANDS - 45, 1, 32)
    model.add(keras.layers.Conv2D(filters=32,
                                  kernel_size=(1, 16),
                                  padding='same',
                                  data_format="channels_last",
                                  activation='tanh'))
    
    # Flatten before dense layer
    model.add(keras.layers.Flatten())
    
    # Add fully connected (4)
    # -----------------------
    # Intput_shape (batch, rows, cols, channels) = (-, (BANDS - 45) x 1 x 32)
    # Output_shape (batch, dim) = (-, 800)
    model.add(keras.layers.Dense(units=800,
                                 activation='tanh'))
    
    # Add fully connected (5)
    # -----------------------
    # Intput_shape (batch, dim) = (-, 800)
    # Output_shape (batch, dim) = (-, 800)
    model.add(keras.layers.Dense(units=800,
                                 activation='softmax'))
    
    # Add fully connected to reduce to number of categories
    # -----------------------------------------------------
    # Intput_shape (batch, dim) = (-, 800)
    # Output_shape (batch, dim) = (-, CLASSES)
    model.add(keras.layers.Dense(units=CLASSES,
                                 activation='softmax'))
    
    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Print the model summary to output file
    #     To print to stdout: model.summary()
    with open(OUTPUT_FILE, 'a') as f:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # Return the model
    return model

def get_model_3(parameters):
    """Generates the model.
    
    Generates a lightgbm model.
    
    """
    # Generate model
    model = lgb.LGBMClassifier(
                    objective = 'multiclass',
                    class_weight = 'balanced',
                    n_estimators = parameters['n_estimators'],
                    min_child_samples = parameters['min_child_samples'])
    
    # Return the model
    return model

def get_cv_model_3(parameters):
    """Generates the cross validation model.
    
    Generates a lightgbm cross validation model for each n_estimators in
    N_ESTIMATORS_LIST.
    
    """
    # Generate model
    model = lgb.LGBMClassifier(objective = 'multiclass',
                               class_weight = 'balanced')
    
    # Cross validation parameters
    param_grid = {
        'n_estimators': N_ESTIMATORS_LIST,
        'min_child_samples' : MIN_CHILD_SAMPLES_LIST
    }
    
    # Grid search
    cv_model = GridSearchCV(model, param_grid, cv=3)
    
    # Return the models
    return cv_model

def get_model_4(parameters):
    """Generates the model.
    
    Generates a SVM model.
    
    """
    # Generate model
    model = svm.SVC(C=40.0, gamma='scale', probability=True,
                    class_weight='balanced',
                    decision_function_shape='ovr')
    
    # Return the model
    return model

def get_model_5(parameters):
    """Generates the model.
    
    Generates a SVM model.
    
    """
    # Generate model
    model = svm.LinearSVC(max_iter=100000)
    
    # Return the model
    return model

def get_model_6(parameters):
    """Generates the model.
    
    Generates a RandomForestClassifier model.
    
    """
    # Generate model
    n_estimators = parameters['num_classes'] * parameters['n_estimators']
    model = RandomForestClassifier(
                    class_weight='balanced',
                    n_estimators=n_estimators,
                    min_samples_split=parameters['min_child_samples'])
    
    # Return the model
    return model

def get_model(parameters):
    """Generates the model.
    
    Chooses the generator depending on the selected model.
    
    """
    if MODEL == 6:
        return get_model_6(parameters)
    elif MODEL == 5:
        return get_model_5(parameters)
    elif MODEL == 4:
        return get_model_4(parameters)
    elif MODEL == 3:
        if CROSS_VALIDATION:
            return get_cv_model_3(parameters)
        else:
            return get_model_3(parameters)
    elif MODEL == 2:
        return get_model_2(parameters)
    else:
        return get_model_1(parameters)

def load_keras_model(trained_models_dir, image_name):
    """Loads keras trained models."""
    # Load the model
    model_file = os.path.join(trained_models_dir,
                              "{}_model.h5py".format(image_name))
    model = keras.models.load_model(model_file)
    
    # Return the model
    return model

def load_lgb_model(trained_models_dir, image_name):
    """Loads lightgbm trained models."""
    # Load the model
    model_file = os.path.join(trained_models_dir,
                              "{}_model.txt".format(image_name))
    model = lgb.Booster(model_file=model_file)
    
    # Return the model
    return model

def load_joblib_model(trained_models_dir, image_name):
    """Loads scikit learn trained models."""
    # Load the model
    model_file = os.path.join(trained_models_dir,
                              "{}_model.joblib".format(image_name))
    model = joblib.load(model_file)
    
    # Return the model
    return model

def load_model(model, trained_models_dir, image_name):
    """Loads the trained model."""
#    if model == "keras":
    if model == 1:
        return load_keras_model(trained_models_dir, image_name)
#    elif model == "lgb":
    elif model == 3:
        return load_lgb_model(trained_models_dir, image_name)
#    elif model = "sklearn":
    else:
        return load_joblib_model(trained_models_dir, image_name)

def train_model_1(model, X_train, y_train, X_val, y_val, image_name):
    """Trains the model.
    
    Model from the paper `Deep Convolutional Neural Networks for
    Hyperspectral Image Classification`.
    
    """
    # Train the model
    batch_size = 50
    start_time = timeit.default_timer()
    history = model.fit(X_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=TRAIN_EPOCHS,
                        verbose=1,
                        validation_data=(X_val, y_val))
    end_time = timeit.default_timer()
    time = end_time - start_time
    with open(OUTPUT_FILE, 'a') as f:
        f.write("\ntraining time:   {:.3f}s\n".format(time))
    
    # Save the model
    model_file = os.path.join(OUTPUT_DIR, "{}_model.h5py".format(image_name))
    model.save(model_file)
    
    return model, history

def train_model_2(model, X_train, y_train, X_val, y_val, image_name):
    """Trains the model.
    
    Model from the paper `Hyperspectral Image Classification with
    Convolutional Neural Networks`.
    
    """
    # Train the model
    batch_size = 50
    start_time = timeit.default_timer()
    history = model.fit(X_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=TRAIN_EPOCHS,
                        verbose=1,
                        validation_data=(X_val, y_val))
    end_time = timeit.default_timer()
    time = end_time - start_time
    with open(OUTPUT_FILE, 'a') as f:
        f.write("\ntraining time:   {:.3f}s\n".format(time))
    
    # Save the model
    model_file = os.path.join(OUTPUT_DIR, "{}_model.h5py".format(image_name))
    model.save(model_file)
    
    return model, history

def train_model_3(model, X_train, y_train, X_val, y_val, image_name):
    """Trains the model.
    
    Trains a lightgbm model.
    
    """
    # Train the model
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    
    # Save the model
    model_file = os.path.join(OUTPUT_DIR, "{}_model.txt".format(image_name))
    model.booster_.save_model(model_file)
    
    return model, model

def train_cv_model_3(cv_model, X_train, y_train, X_val, y_val, image_name):
    """Trains the model.
    
    Trains a lightgbm model for each n_estimators in N_ESTIMATORS_LIST.
    
    """
    # Cross validation with train and validation sets
    X_cv = np.append(X_train, X_val, axis=0)
    y_cv = np.append(y_train, y_val, axis=0)
    cv_model.fit(X_cv, y_cv)
    
    # Write best parameters
    with open(OUTPUT_FILE, 'a') as f:
        f.write("\ncv_best_params: {}\n".format(cv_model.best_params_))
    
    # Save the model
    model_file = os.path.join(OUTPUT_DIR, "{}_cv_model.txt".format(image_name))
    cv_model.best_estimator_.booster_.save_model(model_file)
    
    return cv_model.best_estimator_, cv_model.best_estimator_

def train_model_4(model, X_train, y_train, image_name):
    """Trains the model.
    
    Trains a SVM model.
    
    """
    # Train the model
    model.fit(X_train, y_train)
    
    # Save the model
    model_file = os.path.join(OUTPUT_DIR,
                              "{}_model.joblib".format(image_name))
    joblib.dump(model, model_file)
    
    return model, model

def train_model(model, X_train, y_train, X_val, y_val, image_name):
    """Trains the model.
    
    Chooses the training function depending on the selected model.
    
    """
    if MODEL == 1:
        return train_model_1(model, X_train, y_train, X_val, y_val, image_name)
    elif MODEL == 3:
        if CROSS_VALIDATION:
            return train_cv_model_3(model, X_train, y_train,
                                    X_val, y_val, image_name)
        else:
            return train_model_3(model, X_train, y_train,
                                 X_val, y_val, image_name)
    elif MODEL == 2:
        return train_model_2(model, X_train, y_train, X_val, y_val, image_name)
    else:
        # For models 4, 5 and 6
        return train_model_4(model, X_train, y_train, image_name)

def plot_eval_1(trained_model, image_name):
    """Plots the training evaluation data of the model.
    
    Model from the paper `Deep Convolutional Neural Networks for
    Hyperspectral Image Classification`.
    
    """
    # Get training evaluation data
    train_accuracy = trained_model.history['acc']
    train_val_accuracy = trained_model.history['val_acc']
    train_loss = trained_model.history['loss']
    train_val_loss = trained_model.history['val_loss']
    
    # Generate accuracy plot
    epochs = range(len(train_accuracy))
    plt.figure()
    plt.plot(epochs, train_accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, train_val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    # Save accuracy plot
    plot_file = os.path.join(OUTPUT_DIR,
                             "{}_training_accuracy".format(image_name))
    plt.savefig(plot_file + ".svg", bbox_inches='tight', format='svg')
    
    # Generate loss plot
    plt.figure()
    plt.plot(epochs, train_loss, 'bo', label='Training loss')
    plt.plot(epochs, train_val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    # Save loss plot
    plot_file = os.path.join(OUTPUT_DIR, "{}_training_loss".format(image_name))
    plt.savefig(plot_file + ".svg", bbox_inches='tight', format='svg')

def plot_eval_2(trained_model, image_name):
    """Plots the training evaluation data of the model.
    
    Model from the paper `Hyperspectral Image Classification with
    Convolutional Neural Networks`.
    
    """
    # Get training evaluation data
    train_accuracy = trained_model.history['acc']
    train_val_accuracy = trained_model.history['val_acc']
    train_loss = trained_model.history['loss']
    train_val_loss = trained_model.history['val_loss']
    
    # Generate accuracy plot
    epochs = range(len(train_accuracy))
    plt.figure()
    plt.plot(epochs, train_accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, train_val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    # Save accuracy plot
    plot_file = os.path.join(OUTPUT_DIR,
                             "{}_training_accuracy".format(image_name))
    plt.savefig(plot_file + ".svg", bbox_inches='tight', format='svg')
    
    # Generate loss plot
    plt.figure()
    plt.plot(epochs, train_loss, 'bo', label='Training loss')
    plt.plot(epochs, train_val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    # Save loss plot
    plot_file = os.path.join(OUTPUT_DIR, "{}_training_loss".format(image_name))
    plt.savefig(plot_file + ".svg", bbox_inches='tight', format='svg')

def plot_eval_3(trained_model, X_val, y_val, image_name):
    """Plots the training evaluation data of the model.
    
    Plots the training evaluation data of a lightgbm model.
    
    """
    # FOR EACH CLASS
    # val_pred = trained_model.predict_proba(X_val, num_iteration=iteration)
    
    iterations = trained_model.booster_.current_iteration()
#    results = np.zeros((2, iterations))
    results = np.zeros((iterations,))
    for pos in range(iterations):
        
        # Calculate the current iteration (from 1 to iterations)
        iteration = pos + 1
        
        # Predict validation set for the current iteration
#        start_time = timeit.default_timer()
        val_pred = trained_model.predict(X_val, num_iteration=iteration)
#        end_time = timeit.default_timer()
#        time = end_time - start_time
#        speed = int(X_val.shape[0] / time)
        
        # Number of hits
        val_ok = (val_pred == y_val)
        
        # Percentage of hits
        val_acc = val_ok.sum() / val_ok.size
        
        # Actualize data for plotting results
#        results[0][pos] = time
#        results[1][pos] = val_acc
        results[pos] = val_acc
    
    # Generate accuracy plot
    plt.figure()
#    plt.plot(results[0], results[1], 'b')
    plt.plot(results, 'b')
    plt.title('Validation accuracy')
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.legend()
    
    # Save validation plot
    plot_file = os.path.join(OUTPUT_DIR, "{}_val_accuracy".format(image_name))
    plt.savefig(plot_file + ".svg", bbox_inches='tight', format='svg')

def plot_eval_4(trained_model, X_val, y_val, image_name):
    """Plots the training evaluation data of the model.
    
    Model from the paper `Hyperspectral Image Classification with
    Convolutional Neural Networks`.
    
    """
    # Predict with validation data
    start_time = timeit.default_timer()
    trained_model.predict(X_val)
    end_time = timeit.default_timer()
    time = end_time - start_time
    speed = int(X_val.shape[0] / time)
    
    # Get loss and accuracy
    val_accuracy = trained_model.score(X_val, y_val)
    
    # Prepare results messages
    msg_time = "\nprediction time: {:.3f}s ({}px/s)\n".format(time, speed)
    msg_val_acc = "val_accuracy:   {:.3f}\n\n".format(val_accuracy)
    
    # Write results messages
    with open(OUTPUT_FILE, 'a') as f:
        f.write(msg_time)
        f.write(msg_val_acc)

def plot_eval(trained_model, X_val, y_val, image_name):
    """Plots the training evaluation data of the model.
    
    Chooses the plot function depending on the selected model.
    
    """
    if MODEL == 1:
        return plot_eval_1(trained_model, image_name)
    elif MODEL == 3:
        if not CROSS_VALIDATION:
            return plot_eval_3(trained_model, X_val, y_val, image_name)
    elif MODEL == 2:
        return plot_eval_2(trained_model, image_name)
    else:
        # For models 4, 5 and 6
        return plot_eval_4(trained_model, X_val, y_val, image_name)

def predict_1(trained_model, X_test, y_test):
    """Predicts the test evaluation data of the model.
    
    Model from the paper `Deep Convolutional Neural Networks for
    Hyperspectral Image Classification`.
    
    """
    # Predict with test data
    start_time = timeit.default_timer()
    test_prediction = trained_model.predict(X_test)
    end_time = timeit.default_timer()
    time = end_time - start_time
    speed = int(X_test.shape[0] / time)
    
    # Get loss and accuracy
    test_loss, test_accuracy = trained_model.evaluate(X_test, y_test)
    
    # Prepare results messages
    msg_time = "prediction time: {:.3f}s ({}px/s)\n".format(time, speed)
    msg_test_loss = "test_loss:       {:.3f}\n".format(test_loss)
    msg_test_acc = "test_accuracy:   {:.3f}\n\n".format(test_accuracy)
    
    # Write results messages
    with open(OUTPUT_FILE, 'a') as f:
        f.write(msg_time)
        f.write(msg_test_loss)
        f.write(msg_test_acc)

def predict_2(trained_model, X_test, y_test):
    """Predicts the test evaluation data of the model.
    
    Model from the paper `Hyperspectral Image Classification with
    Convolutional Neural Networks`.
    
    """
    # Predict with test data
    start_time = timeit.default_timer()
    test_prediction = trained_model.predict(X_test)
    end_time = timeit.default_timer()
    time = end_time - start_time
    speed = int(X_test.shape[0] / time)
    
    # Get loss and accuracy
    test_loss, test_accuracy = trained_model.evaluate(X_test, y_test)
    
    # Prepare results messages
    msg_time = "prediction time: {:.3f}s ({}px/s)\n".format(time, speed)
    msg_test_loss = "test_loss:       {:.3f}\n".format(test_loss)
    msg_test_acc = "test_accuracy:   {:.3f}\n\n".format(test_accuracy)
    
    # Write results messages
    with open(OUTPUT_FILE, 'a') as f:
        f.write(msg_time)
        f.write(msg_test_loss)
        f.write(msg_test_acc)

def predict_3(trained_model, X_test, y_test, image_name):
    """Predicts the test evaluation data of the model.
    
    Predicts the test evaluation data of a lightgbm model.
    
    """
    iterations = trained_model.booster_.current_iteration()
    results = np.zeros((iterations, ))
    for pos in range(iterations):
        
        # Calculate the current iteration (from 1 to iterations)
        iteration = pos + 1
        
        # Predict test set for the current iteration
        start_time = timeit.default_timer()
        test_pred = trained_model.predict(X_test, num_iteration=iteration)
        end_time = timeit.default_timer()
        time = end_time - start_time
        speed = int(X_test.shape[0] / time)
        
        # Number of hits
        test_ok = (test_pred == y_test)
        
        # Percentage of hits
        test_acc = test_ok.sum() / test_ok.size
        
        # Actualize data for plotting results
        results[pos] = test_acc
        
        # Prepare test messages
        classes = trained_model.n_classes_
        msg_estimators = "\nn_estimators:      {}\n".format(iteration)
        msg_trees = "Number of trees:   {}\n".format(classes * iteration)
        msg_time = "prediction time:   {:.3f}s ({}px/s)\n".format(time, speed)
        msg_test_acc = "test_acc:          {:.3f}\n".format(test_acc)
        
        # Write test messages
        with open(OUTPUT_FILE, 'a') as f:
            f.write(msg_estimators)
            f.write(msg_trees)
            f.write(msg_time)
            f.write(msg_test_acc)
    
    # Generate accuracy plot
    plt.figure()
    plt.plot(results, 'b')
    plt.title('Test accuracy')
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.legend()
    
    # Save test plot
    plot_file = os.path.join(OUTPUT_DIR, "{}_test_accuracy".format(image_name))
    plt.savefig(plot_file + ".svg", bbox_inches='tight', format='svg')

def cv_predict_3(trained_model, X_test, y_test):
    """Predicts the test evaluation data of the model.
    
    Predicts the test evaluation data of a lightgbm model for each
    n_estimators in N_ESTIMATORS_LIST.
    
    """
    # Predict test set
    start_time = timeit.default_timer()
    test_pred = trained_model.predict(X_test)
    end_time = timeit.default_timer()
    time = end_time - start_time
    speed = int(X_test.shape[0] / time)
    
    # Number of hits
    test_ok = (test_pred == y_test)
    
    # Percentage of hits
    test_acc = test_ok.sum() / test_ok.size
    
    # Prepare test messages
    classes = trained_model.n_classes_
    msg_time = "prediction time: {:.3f}s ({}px/s)\n".format(time,
                                                            speed)
    msg_test_acc = "test_acc:        {:.3f}\n".format(test_acc)
    
    # Write test messages
    with open(OUTPUT_FILE, 'a') as f:
        f.write(msg_time)
        f.write(msg_test_acc)

def predict_4(trained_model, X_test, y_test):
    """Predicts the test evaluation data of the model.
    
    SVM Model.
    
    """
    # Predict with test data
    start_time = timeit.default_timer()
    trained_model.predict(X_test)
    end_time = timeit.default_timer()
    time = end_time - start_time
    speed = int(X_test.shape[0] / time)
    
    # Get loss and accuracy
    test_accuracy = trained_model.score(X_test, y_test)
    
    # Prepare results messages
    msg_time = "prediction time: {:.3f}s ({}px/s)\n".format(time, speed)
    msg_test_acc = "test_accuracy:   {:.3f}\n\n".format(test_accuracy)
    
    # Write results messages
    with open(OUTPUT_FILE, 'a') as f:
        f.write(msg_time)
        f.write(msg_test_acc)

def predict(trained_model, X_test, y_test, image_name):
    """Predicts the test evaluation data of the model.
    
    Chooses the prediction function depending on the selected model.
    
    """
    if MODEL == 1:
        return predict_1(trained_model, X_test, y_test)
    elif MODEL == 3:
        if CROSS_VALIDATION:
            return cv_predict_3(trained_model, X_test, y_test)
        else:
            return predict_3(trained_model, X_test, y_test, image_name)
    elif MODEL == 2:
        return predict_2(trained_model, X_test, y_test)
    else:
        # For models 4, 5 and 6
        return predict_4(trained_model, X_test, y_test)

def class_predict_3(trained_model, X_test, y_test, image_name):
    """Predicts the test evaluation data of the model.
    
    Predicts the test evaluation data of a lightgbm model.
    
    """
    # Predict test set
    try:
        test_pred = trained_model.predict_proba(X_test)
    except:
        test_pred = trained_model.predict(X_test)
    
    if len(test_pred.shape) == 1:
        raise Exception("Probabilistic prediction needed.")
    
    # Transform y_test
    if len(y_test.shape) > 1:
        y_test = np.argmax(y_test, axis=1)
    
    classes = np.unique(y_test)
    results = np.zeros((len(classes), ))
    for class_num in classes:
        
        # Take predictions for current class
        X_pred = test_pred[y_test == class_num, :]
        
        # Number of hits
        pred_ok = (np.argmax(X_pred, axis=1) == class_num).sum()
        
        # Percentage of hits
        pred_acc = pred_ok / X_pred.shape[0]
        
        # Actualize data for plotting results
        results[class_num] = pred_acc
        
        # Write test message
        with open(OUTPUT_FILE, 'a') as f:
            f.write("test_acc of class {}: {:.3f}\n".format(class_num,
                                                            pred_acc))
    
    # Generate accuracy plot
    plt.figure()
    plt.bar(classes, results, align='center')
    plt.xticks(classes, classes)
    plt.title('Per class test accuracy')
    plt.xlabel('class')
    plt.ylabel('accuracy')
    plt.legend()
    
    # Save test plot
    plot_file = os.path.join(OUTPUT_DIR, "{}_test_accuracy".format(image_name))
    plt.savefig(plot_file + ".svg", bbox_inches='tight', format='svg')

def class_predict(trained_model, X_test, y_test, image_name):
    """Predicts the test evaluation data of the model.
    
    Chooses the prediction function depending on the selected model.
    
    """
    if MODEL == 1:
        return class_predict_3(trained_model, X_test, y_test, image_name)
    elif MODEL == 3:
        return class_predict_3(trained_model, X_test, y_test, image_name)
    elif MODEL == 2:
        return class_predict_2(trained_model, X_test, y_test)
    else:
        # For models 4, 5 and 6
        return class_predict_3(trained_model, X_test, y_test, image_name)

def join_prediction(model1, model2, X_test1, X_test2, y_test, image_name):
    # Predict test set
    try:
        test_pred1 = model1.predict_proba(X_test1)
    except:
        test_pred1 = model1.predict(X_test1)
    
    try:
        test_pred2 = model2.predict_proba(X_test2)
    except:
        test_pred2 = model2.predict(X_test2)
    
    test_pred3 = test_pred1 + test_pred2
    
    # y_test to label encoding
    if len(y_test.shape) > 1:
        y_test = np.argmax(y_test, axis=1)
    
    # Number of hits
    pred1_ok = (np.argmax(test_pred1, axis=1) == y_test).sum()
    pred2_ok = (np.argmax(test_pred2, axis=1) == y_test).sum()
    pred3_ok = (np.argmax(test_pred3, axis=1) == y_test).sum()
    
    # Percentage of hits
    pred1_acc = pred1_ok / y_test.shape[0]
    pred2_acc = pred2_ok / y_test.shape[0]
    pred3_acc = pred3_ok / y_test.shape[0]
    
    # Write test message
    with open(OUTPUT_FILE, 'a') as f:
        f.write("test_acc of model1: {:.3f}\n".format(pred1_acc))
        f.write("test_acc of model2: {:.3f}\n".format(pred2_acc))
        f.write("test_acc combined: {:.3f}\n".format(pred3_acc))
    
#    classes = np.unique(test_classes)
#    results = np.zeros((len(classes), ))
#    for class_num in classes:
#        
#        # Take predictions for current class
#        X_pred = test_pred[test_classes == class_num, :]
#        
#        # Number of hits
#        pred_ok = (np.argmax(X_pred, axis=1) == class_num).sum()
#        
#        # Percentage of hits
#        pred_acc = pred_ok / X_pred.shape[0]
#        
#        # Actualize data for plotting results
#        results[class_num] = pred_acc
#        
#        # Write test message
#        with open(OUTPUT_FILE, 'a') as f:
#            f.write("test_acc of class {}: {:.3f}\n".format(class_num,
#                                                            pred_acc))
#    
#    # Generate accuracy plot
#    plt.figure()
#    plt.bar(classes, results, align='center')
#    plt.xticks(classes, classes)
#    plt.title('Per class test accuracy')
#    plt.xlabel('class')
#    plt.ylabel('accuracy')
#    plt.legend()
#    
#    # Save test plot
#    plot_file = os.path.join(OUTPUT_DIR,
#                             "{}_test_accuracy".format(image_name))
#    plt.savefig(plot_file + ".svg", bbox_inches='tight', format='svg')

def join_all_predictions(model1, model3, model4, model6,
                         X_test1, X_test2, y_test, image_name):
    # Predict test set
    try:
        test_pred1 = model1.predict_proba(X_test1)
    except:
        test_pred1 = model1.predict(X_test1)
    
    try:
        test_pred3 = model3.predict_proba(X_test2)
    except:
        test_pred3 = model3.predict(X_test2)
    
    try:
        test_pred4 = model4.predict_proba(X_test2)
    except:
        test_pred4 = model4.predict(X_test2)
    
    try:
        test_pred6 = model6.predict_proba(X_test2)
    except:
        test_pred6 = model6.predict(X_test2)
    
    test_pred = test_pred1 + test_pred3 + test_pred4 + test_pred6
    
    # y_test to label encoding
    if len(y_test.shape) > 1:
        y_test = np.argmax(y_test, axis=1)
    
    # Number of hits
    pred1_ok = (np.argmax(test_pred1, axis=1) == y_test).sum()
    pred3_ok = (np.argmax(test_pred3, axis=1) == y_test).sum()
    pred4_ok = (np.argmax(test_pred4, axis=1) == y_test).sum()
    pred6_ok = (np.argmax(test_pred6, axis=1) == y_test).sum()
    pred_ok = (np.argmax(test_pred, axis=1) == y_test).sum()
    
    # Percentage of hits
    pred1_acc = pred1_ok / y_test.shape[0]
    pred3_acc = pred3_ok / y_test.shape[0]
    pred4_acc = pred4_ok / y_test.shape[0]
    pred6_acc = pred6_ok / y_test.shape[0]
    pred_acc = pred_ok / y_test.shape[0]
    
    # Write test message
    with open(OUTPUT_FILE, 'a') as f:
        f.write("test_acc of model1: {:.3f}\n".format(pred1_acc))
        f.write("test_acc of model3: {:.3f}\n".format(pred3_acc))
        f.write("test_acc of model4: {:.3f}\n".format(pred4_acc))
        f.write("test_acc of model6: {:.3f}\n".format(pred6_acc))
        f.write("test_acc combined: {:.3f}\n".format(pred_acc))

def main(argv):
    
    # Management of arguments
    args = parse_args()
    use_args(args)
    
    # Create the output dir
    try:
        os.makedirs(OUTPUT_DIR)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print("The directory already exists.")
            answer = raw_input("Do you want to continue? (yes, no)").lower()
            if answer not in ["y", "yes"]:
                sys.exit(0)
        else:
            raise
    
    # Create the output file
    with open(OUTPUT_FILE, 'w') as f:
        if JOIN_MODELS:
            f.write("Models to join: {} and {}\n\n".format(MODELS[MODEL1],
                                                           MODELS[MODEL2]))
        else:
            f.write("Selected model: {}\n\n".format(MODELS[MODEL]))
    
    # Get images information
    with open(IMAGES_FILE, 'r') as f:
        images_information = json.loads(f.read())
    
    for image_info in images_information.itervalues():
        
        if ALL_TOGETHER:
            
            # Get and preprocess image for model 1
            (_, _, X_test1,
             _, _, y_test,
             _, _, _) = preprocess_image(image_info, DATA_PATH,
                                         OUTPUT_DIR, OUTPUT_FILE,
                                         indexes_dir=INDEXES_DIR,
                                         keras_cnn=True,
                                         features=FEATURES)
            
            # Get and preprocess image for model 3, 4 and 6
            (_, _, X_test2,
             _, _, _,
             _, _, _) = preprocess_image(image_info, DATA_PATH,
                                         OUTPUT_DIR, OUTPUT_FILE,
                                         indexes_dir=INDEXES_DIR,
                                         keras_cnn=False,
                                         features=FEATURES)
            
            # Load the models
            model1 = load_model(1, TRAINED_MODELS_DIRS[0], image_info['key'])
            model3 = load_model(3, TRAINED_MODELS_DIRS[1], image_info['key'])
            model4 = load_model(4, TRAINED_MODELS_DIRS[2], image_info['key'])
            model6 = load_model(6, TRAINED_MODELS_DIRS[3], image_info['key'])
            
            # Test the models
            join_all_predictions(model1, model3, model4, model6,
                                 X_test1, X_test2, y_test, image_info['key'])
        
        elif JOIN_MODELS:
            
            # Get and preprocess image for the first model
            (_, _, X_test1,
             _, _, y_test,
             _, _, _) = preprocess_image(image_info, DATA_PATH,
                                         OUTPUT_DIR, OUTPUT_FILE,
                                         indexes_dir=INDEXES_DIR,
                                         keras_cnn=(MODEL1 == 1),
                                         features=FEATURES)
            
            # Get and preprocess image for the second model
            (_, _, X_test2,
             _, _, _,
             _, _, _) = preprocess_image(image_info, DATA_PATH,
                                         OUTPUT_DIR, OUTPUT_FILE,
                                         indexes_dir=INDEXES_DIR,
                                         keras_cnn=(MODEL2 == 1),
                                         features=FEATURES)
            
            # Load the models
            model1 = load_model(MODEL1, TRAINED_MODELS_DIR, image_info['key'])
            
            model2 = load_model(MODEL2, TRAINED_MODELS_DIR2, image_info['key'])
            
            # Test the models
            join_prediction(model1, model2, X_test1, X_test2,
                            y_test, image_info['key'])
        
        else:
            
            # Get and preprocess image
            (X_train, X_val, X_test,
             y_train, y_val, y_test,
             num_pixels, num_features,
             num_classes) = preprocess_image(image_info, DATA_PATH,
                                             OUTPUT_DIR, OUTPUT_FILE,
                                             indexes_dir=INDEXES_DIR,
                                             keras_cnn=(MODEL == 1),
                                             features=FEATURES)
            
            # Prepare model training parameters
            parameters = model_parameters(num_features, num_classes,
                                          image_info)
            
            if TRAINED_MODELS:
                
                # Load the model
                model = load_model(MODEL, TRAINED_MODELS_DIR,
                                   image_info['key'])
                
                SAVE_PREDICTION = False
                if SAVE_PREDICTION: # TODO
                    
                    # Save prediction
                    save_prediction(model, X_test, y_test, image_info['key'])
                
                else:
                    
                    # Test the model
                    class_predict(model, X_test, y_test, image_info['key'])
            
            else:
                
                # Generate the model
                model = get_model(parameters)
                
                # Train the model
                model, history = train_model(model, X_train, y_train,
                                             X_val, y_val, image_info['key'])
                
                # Plot training evaluation data
                plot_eval(history, X_val, y_val, image_info['key'])
                
                # Prediction
                predict(model, X_test, y_test, image_info['key'])

if __name__ == "__main__":
    main(sys.argv[1:])


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module for hyperspectral images preprocessing."""
from __future__ import division, absolute_import, print_function
import os
import scipy.io
import numpy as np

import keras

def load_image(image_info, data_path, output_file):
    """Loads the image and the ground truth from a `mat` file.
    
    Parameters
    ----------
    image_info: dict
        Dict structure with information of the image.
    data_path: String
        Absolute path of the hyperspectral images directory.
    output_file: String
        Absolute path of the output file.
    
    Returns
    -------
    out: NumPy array, NumPy array
        The image and the ground truth data.
    
    """
    # Image name
    image_name = image_info['key']
    
    # Filenames
    input_file = os.path.join(data_path, image_info['file'])
    label_file = os.path.join(data_path, image_info['file_gt'])
    
    try:
        
        # Load image message
        with open(output_file, 'a') as f:
            f.write("=" * 65)
            f.write("\n\nLoading image {} ...\n".format(image_name))
        
        # Load image and ground truth files
        X = scipy.io.loadmat(input_file)[image_name]
        y = scipy.io.loadmat(label_file)[image_info['key_gt']]
    
    except:
        
        # Download image message
        with open(output_file, 'a') as f:
            f.write("Image files not found.\n")
            f.write("Downloading: {} ...\n".format(image_info['url']))
            f.write("Downloading: {} ...\n".format(image_info['url_gt']))
        
        # Download image and ground truth files
        os.system("wget {} -O {}".format(image_info['url'], input_file))
        os.system("wget {} -O {}".format(image_info['url_gt'], label_file))
	    
        # Load image message
        with open(output_file, 'a') as f:
            f.write("Loading image {} ...\n".format(image_name))
        
	    # Load image and ground truth files
        X = scipy.io.loadmat(input_file)[image_name]
        y = scipy.io.loadmat(label_file)[image_info['key_gt']]
    
    return X, y, image_name

def normalize(X, a, b):
    """Normalizes float data between two values.
    
    Parameters
    ----------
    X: NumPy array
        NumPy array of floats to normalize.
    a: float
        Minimum value of the output range.
    b: float
        Maximum value of the output range.
    
    Returns
    -------
    out: NumPy array
        `X` normalized between `a` and `b`.
    
    """
    return (b-a) * (X-X.min()) / (X.max()-X.min()) + a

def pixel_classification_preprocessing(X, y, output_file, image_info,
                                       normalization=False, features=0):
    """Preprocesses hyperspectral images for pixel classification.
    
    Reshapes the image and the ground truth data, keeps only the labeled
    pixels, normalizes if necesary, and rename the classes to ordered
    integers from 0.
    
    Parameters
    ----------
    X: NumPy array
        The image data.
    y: NumPy array
        The ground truth data.
    output_file: String
        Absolute path of the output file.
    image_info: dict
        Dict structure with information of the image.
    normalization: bool, optional
        Flag to activate data normalization.
    features: int, optional
        Nuber of best features to use. If `0` (default) it uses every
        feature.
    
    Returns
    -------
    out: NumPy array, NumPy array, int, int, int
        The pixels and labels data prepreocessed and the remaining
        number of pixels, features and classes respectively.
    
    """
    # Preprocessing message
    with open(output_file, 'a') as f:
        f.write("\nPreprocessing image {} ...\n".format(image_info['key']))
    
    # Reshape them to ignore spatiality
    X = X.reshape(-1, X.shape[2])
    y = y.reshape(-1)
    
    if features > 0:
        # Best features selection
        X = X[:, sorted(image_info['features'][0:features])]
    
    # Keep only labeled pixels
    X = X[y > 0, :]
    y = y[y > 0]
    
    # Rename clases to ordered integers from 0
    for new_class_num, old_class_num in enumerate(np.unique(y)):
        y[y == old_class_num] = new_class_num
    
    if normalization:
        
        # Normalize data to range [-1.0, 1.0]
        X = normalize(X, -1.0, 1.0)
    
    # Get image characteristics
    num_pixels, num_features = X.shape
    num_classes = len(np.unique(y))
    
    # Write image characteristics messages
    with open(output_file, 'a') as f:
        f.write("num_class:    {}\n".format(num_classes))
        f.write("num_features: {}\n".format(num_features))
        f.write("Number of pixels:      {}\n\n".format(num_pixels))
    
    return X, y, num_pixels, num_features, num_classes

def shuffle(X, y, output_dir, file_name, indexes_dir=None):
    """Shuffle the data and labels.
    
    Parametes
    ---------
    X: NumPy array
        Pixels to shuffle.
    y: NumPy array
        Labels to shuffle.
    output_dir: String
        Absolute path of the output directory.
    file_name: String
        Name of the input and output index files.
    indexes_dir: None | String, optional
        If it exists, absolute path of the indexes directory.
    
    Returns
    -------
    out: NumPy array, NumPy array
        Shuffled pixels and labels.
    
    """
    if indexes_dir:
        
        # Load the index file
        index = np.load(os.path.join(indexes_dir, file_name))
    
    else:
        
        # Generate the index
        index = np.random.permutation(X.shape[0])
    
    # Save random index for reproducibility
    np.save(os.path.join(output_dir, file_name), index)
    
    # Return shuffled data
    return X[index], y[index]

def separate_pixels(X, y, image_info, output_dir, output_file,
                    image_name, indexes_dir=None):
    """Separate pixels and labels into train, validation and test sets.
    
    Input data has to be preprocessed so classes are consecutively
    named from '0'.
    
    Parameters
    ----------
    X: NumPy array
        The preprocessed pixels.
    y: NumPy array
        The preprocessed labels.
    image_info: dict
        Dict structure with information of the image.
    output_dir: String
        Absolute path of the output directory.
    output_file: String
        Absolute path of the output file.
    image_name: String
        Image name.
    indexes_dir: None | String, optional
        If it exists, absolute path of the indexes directory.
    
    Returns
    -------
    out: (NumPy array, NumPy array, NumPy array,
          NumPy array, NumPy array, NumPy array)
        Structures corresponding to:
            (Train pixels, validation pixels, test pixels,
             train labels, validation labels, test labels)
    
    """
    # Train, validation and test sets message
    with open(output_file, 'a') as f:
        f.write("Generating train, validation and test sets ...\n")
    
    # Get the data sets sizes
    train_pixels = image_info['train_20'][1:]
    val_pixels = image_info['val_20'][1:]
    test_pixels = image_info['test_20'][1:]
    
    # Calculate sizes of each structure
    num_train_pixels = sum(train_pixels)
    num_val_pixels = sum(val_pixels)
    num_test_pixels = sum(test_pixels)
    
    # Shape of each pixel (some models use complex structures for spaciality)
    pixel_shape = X.shape[1:]
    
    # Prepare structures for train, validation and test data
    X_train = np.zeros((num_train_pixels,) + pixel_shape)
    y_train = np.zeros((num_train_pixels,), dtype=int)
    X_val = np.zeros((num_val_pixels,) + pixel_shape)
    y_val = np.zeros((num_val_pixels,), dtype=int)
    X_test = np.zeros((num_test_pixels,) + pixel_shape)
    y_test = np.zeros((num_test_pixels,), dtype=int)
    
    # Fill train, val and test data structures
    train_end = 0
    val_end = 0
    test_end = 0
    for class_num, (num_train_pixels_class,
                    num_val_pixels_class,
                    num_test_pixels_class) in enumerate(zip(train_pixels,
                                                            val_pixels,
                                                            test_pixels)):
        
        # Get instances of class `class_num`
        class_data = X[y == class_num, :]
        class_labels = y[y == class_num]
        
        # Save train pixels
        train_start = train_end
        train_end = train_start + num_train_pixels_class
        class_start = 0
        class_end = num_train_pixels_class
        X_train[train_start:train_end] = class_data[class_start:class_end]
        y_train[train_start:train_end] = class_labels[class_start:class_end]
        
        # Save val pixels
        val_start = val_end
        val_end = val_start + num_val_pixels_class
        class_start = class_end
        class_end = class_end + num_val_pixels_class
        X_val[val_start:val_end] = class_data[class_start:class_end]
        y_val[val_start:val_end] = class_labels[class_start:class_end]
        
        # Save test pixels
        test_start = test_end
        test_end = test_start + num_test_pixels_class
        class_start = class_end
        class_end = class_end + num_test_pixels_class
        X_test[test_start:test_end] = class_data[class_start:class_end]
        y_test[test_start:test_end] = class_labels[class_start:class_end]
    
    # Shuffle train data
    index_file = "{}_train_index.npy".format(image_name)
    X_train, y_train = shuffle(X_train, y_train, output_dir,
                               index_file, indexes_dir)
    
    # Write characteristics of the generated data sets to the output file
    with open(output_file, 'a') as f:
        f.write("Pixels for training:   {}\n".format(num_train_pixels))
        f.write("Pixels for validating: {}\n".format(num_val_pixels))
        f.write("Pixels for testing:    {}\n".format(num_test_pixels))
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def data_to_cnn_input(X_train, X_val, X_test):
    """Reshape data to fit Keras CNN model input.
    
    Recives train, validation and test pixels and adds extra axes to fit
    Keras CNN model input.
    
    Parameters
    ----------
    X_train: NumPy array
        Train pixels.
    X_val: NumPy array
        Validation pixels.
    X_test: NumPy array
        Test pixels.
    
    Returns
    -------
    out: NumPy array, NumPy array, NumPy array
        Train, validation and test pixels prepared to Keras CNN model.
    
    """
    X_train = X_train[..., np.newaxis, np.newaxis]
    X_val = X_val[..., np.newaxis, np.newaxis]
    X_test = X_test[..., np.newaxis, np.newaxis]
    
    return X_train, X_val, X_test

def labels_to_one_hot(y_train, y_val, y_test):
    """Labels format to one-hot encoding.
    
    Recives train, validation and test labels in label encoding format
    and transforms them into one-hot encoding format.
    
    Parameters
    ----------
    y_train: NumPy array
        Train labels in label encoding format.
    y_val: NumPy array
        Validation labels in label encoding format.
    y_test: NumPy array
        Test labels in label encoding format.
    
    Returns
    -------
    out: NumPy array, NumPy array, NumPy array
        Train, validation and test labels in one-hot encoding format.
    
    """
    # Labels to one-hot encoding
    y_train = keras.utils.to_categorical(y_train)
    y_val = keras.utils.to_categorical(y_val)
    y_test = keras.utils.to_categorical(y_test)
    
    return y_train, y_val, y_test

def preprocess_image(image_info, data_path, output_dir, output_file,
                     indexes_dir=None, keras_cnn=False, features=0):
    """Preprocesses hyperspectral images for pixel classification.
    
    Loads the image pixels from a mat file, preprocesses them and then
    separates the resultant pixels into train, validation and test
    datasets.
    
    Parameters
    ----------
    image_info: dict
        Dict structure with information of the image.
        Its content is:
            'file': image file name
            'file_gt': groud truth file name
            'key': key of the image in te 'mat' file (used as image
                   name)
            'key_gt': key of the ground truth in te 'mat' file
            'url': url of the image file
            'url_gt': url of the groud truth file
            'shape': list containing [rows, cols, features] of the image
            'labels': list with the labels of the ground truth
            'pixels': list with the number of total pixels of each label
            'train_20': list with the number of train pixels per label
                        needed to keep ~20% for training
            'val_20': list with the number of validation pixels per
                      label needed to keep ~20% for training
            'test_20': list with the number of test pixels per label
                       needed to keep ~20% for training
            'train_10': list with the number of train pixels per label
                        needed to keep ~10% for training
            'val_10': list with the number of validation pixels per
                      label needed to keep ~10% for training
            'test_10': list with the number of test pixels per label
                       needed to keep ~10% for training
            'n_estimators': best `number of iterations` parameter
                            selected for trees techniques
            'min_child_samples': best `minimum of data for split`
                                 parameter selected for trees techniques
    data_path: String
        Absolute path of the hyperspectral images directory.
    output_dir: String
        Absolute path of the output directory.
    output_file: String
        Absolute path of the output file.
    indexes_dir: None | String, optional
        If it exists, absolute path of the indexes directory.
    keras_cnn: bool, optional
        Flag to activate the Keras CNN model preprocessing.
    features: int, optional
        Nuber of best features to use. If `0` (default) it uses every
        feature.
    
    Returns
    -------
    out: (NumPy array, NumPy array, NumPy array,
          NumPy array, NumPy array, NumPy array,
          int, int, int)
        Numpy structures corresponding to train pixels, validation
        pixels, test pixels, train labels, validation labels and test
        labels respectively, and integers corresponding to the remaining
        number of pixels features and classes.
    
    """
    # Load image
    X, y, image_name = load_image(image_info, data_path, output_file)
    
    # Image preprocessing for pixel classification
    (X, y,
     num_pixels,
     num_features,
     num_classes) = pixel_classification_preprocessing(X, y,
                                                       output_file,
                                                       image_info,
                                                       normalization=keras_cnn,
                                                       features=features)
    
    # Shuffle the data to avoid spatial information of the original image
    index_file = "{}_random_index.npy".format(image_name)
    X, y = shuffle(X, y, output_dir, index_file, indexes_dir)
    
    # Separate pixels of each class for train, validation and test
    (X_train, X_val, X_test,
     y_train, y_val, y_test) = separate_pixels(X, y, image_info,
                                               output_dir, output_file,
                                               image_name, indexes_dir)
    
    if keras_cnn:
        
        # Reshape data to fit the model
        X_train, X_val, X_test = data_to_cnn_input(X_train, X_val, X_test)
        
        # Labels to one-hot encoding
        y_train, y_val, y_test = labels_to_one_hot(y_train, y_val, y_test)
    
    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            num_pixels, num_features, num_classes)


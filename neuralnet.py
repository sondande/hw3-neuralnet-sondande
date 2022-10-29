"""
Assignment #3: Neural Network

By Sagana Ondande & Ada Ates
"""
import csv
import math
# Import Libraries
import sys
import numpy as np
import pandas as pd
from scipy.special import expit
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

"""
Takes the following parameters:

a. The path to a file containing a data set (e.g., monks1.csv)
b. The number of neurons to use in the hidden layer
c. The learning rate � to use during backpropagation
d. The percentage of instances to use for a training set
e. A random seed as an integer
f. The threshold to use for deciding if the predicted label is a 0 or 1
"""


def data_preprocessing(dataset):
    # Determine whether a column contains numerical or nominial values
    # Create a new Pandas dataframe to maintain order of columns when doing One-Hot Coding on Nominial values
    new_dataframe = pd.DataFrame()
    # Iterate through all the columns of the training_set 
    for x in dataset.columns:
        # Determine if the column 'x' in training set is a Nominial Data or Numerical 
        if is_string_dtype(dataset[x]) and not is_numeric_dtype(dataset[x]):
            # Apply One-Hot Encoding onto Pandas Series at column 'x' 
            dummies = pd.get_dummies(dataset[x], prefix=x, prefix_sep='.', drop_first=True)
            # Combine the One-Hot Encoding Dataframe to our new dataframe to the new_dataframe 
            new_dataframe = pd.concat([new_dataframe, dummies], axis=1)
        else:
            # Find the maximum value in column 'x'
            max_value = max(dataset[x])
            # Find the minimum value in column 'x'
            min_value = min(dataset[x])
            # Check if the column being evaluated is the label column. If so, just add it right into the dataframe
            if x == 'label':
                new_dataframe = pd.concat([new_dataframe, dataset[x]], axis=1)
                continue
            # Ensure we don't run into a zero division error when normalizing all the values
            elif (max_value - min_value) != 0:
                # Apply net value formula to every value in pandas dataframe
                dataset[x] = dataset[x].apply(lambda y: (y - min_value) / (max_value - min_value))
                # Combine New column to our new_dataframe
                new_dataframe = pd.concat([new_dataframe, dataset[x]], axis=1)
    return new_dataframe


"""
Creation of a Sigmoid Function that handles overflow cases as well
"""


def sigmoid(net):
    # If x is a very large positive number, the sigmoid function will be close to 1
    if net >= 0:
        temp = np.exp(-net)
        out = 1 / (1 + temp)
    # If x is a very large negative number, the sigmoid function will be close to 0
    else:
        temp = np.exp(net)
        out = temp / (1 + temp)
    return out


"""
Calculate Net Value for Stochastic Gradient Descent
"""


def net_calculate(weights, x_instance):
    # take the first value from weights as it is part of the net without a corresponding value in the instance
    net = 0
    # Instances and weights are the same length
    for i in range(0, len(x_instance)):
        if i == 0:
            net = weights[0]
        else:
            net += weights[i] * x_instance[i]
    return net

"""
Implementation of Back Propagation Method
"""


def back_propagation(training_set, neural_network):
    # For each instance in the training set:
    for train_i in range(training_set.shape[0]):
        # 1. Feed instance forward through network
        # A. Calculate out_k for each neuron k in the hidden layer
        out_k = []

        # Iterate through all the hiddne layer neurons
        for i in range(number_hidden_neurons):
            out_k.append(sigmoid(net_calculate(neural_network[0][i], training_set[train_i])))

        # B. Calculate out_o for each neuron o, using each out_k as the inputs to neuron o
        # Preappend the value 0 to the beginning of the list. Ensure that the length of both lists are the same.
        # The value is a dummy value to take the place of what the label would have been
        out_o = sigmoid(net_calculate(neural_network[1], [0] + out_k))

        # 2. Calculate the error of the neural network’s prediction
        # A. error = (y - out_o)
        error = training_set[train_i][0] - out_o

        # 3. Calculate feedbacks for the neurons to understand their responsibility in error
        # A. Calculate Feedback_o = out_o * (1 - out_o) * error for output neuron o
        feed_o = error * out_o * (1 - out_o)
        # B. Calculate Feedback_k = out_k * (1 - out_k) * w_k,o * feedback_o for each neuron k in the hidden layer
        feed_k = []

        # Iterate through all the hidden layer neurons
        for i in range(number_hidden_neurons):
            feed_k.append(out_k[i] * (1 - out_k[i]) * neural_network[1][i+1] * feed_o)

        # 4. Update weights based on feedbacks and inputs for all neurons
        # A. Gradient w_k_o = -out_k * feedback_o for each neuron k in the hidden layer
        for i in range(len(neural_network[1])):
            if i == 0:
                # Note difference as for the first weight, we don't include the training input as for x_0, we are having the value of 1, which results in
                # the input being -x_0 = -1
                grad_0 = -1 * feed_o
            else:
                # we do i-1 as we want to take the first output from our hidden layer. Way this is done, we know that is at index i-1 of where i is
                grad_0 = -1 * out_k[i-1] * feed_o
            neural_network[1][i] -= (learning_rate * grad_0)

        # B. Gradient w_i_k = -x_i * feedback_k for each neuron k in the hidden layer
        for i in range(len(neural_network[0])):
            for att in range(len(neural_network[0][0])):
                if att == 0:
                    # Note difference as for the first weight, we don't include the training input as for x_0, we are having the value of 1, which results in
                    # the input being -x_0 = -1
                    grad_k = -1 * feed_k[i]
                else:
                    grad_k = -1 * training_set[train_i][att] * feed_k[i]
                neural_network[0][i][att] -= (learning_rate * grad_k)
    return neural_network


def fit(training_set, validation_set, neural_network, number_of_layers = 0):
    accuracy = 0
    epochs = 0
    while accuracy <= 0.99:
        if epochs == 500:
            break

        back_propagation(training_set, neural_network)
        # Checking against the validation set
        # For each instance in the validation set:
        tt = 0
        tf = 0
        ft = 0
        ff = 0
        for val_i in range(validation_set.shape[0]):
            # 1. Feed instance forward through network
            # A. Calculate out_k for each neuron k in the hidden layer
            out_val_k = []
            # Iterate through all the hidden layer neurons
            for i in range(number_hidden_neurons):
                out_val_k.append(sigmoid(net_calculate(neural_network[0][i], validation_set[val_i])))
            # B. Calculate out_o for each neuron o, using each out_k as the inputs to neuron o
            out_o = sigmoid(net_calculate(neural_network[1], [0] + out_val_k))
            predict = 1 if out_o >= threshold else 0
            instance_label = validation_set[val_i][0]
            # print(f"Val: {val_i}, predicted value: {predicted_val}")
            # print(f"Val: {val_i}, Actual value: {validation_set[val_i][0]}\n")
            # print()
            if predict == 1 and instance_label == 1:
                tt += 1
            elif predict == 1 and instance_label == 0:
                ft += 1
            elif predict == 0 and instance_label == 1:
                tf += 1
            else:
                ff += 1
        accuracy = (tt + ff) / (tt + tf + ft + ff)
        epochs += 1
        print(f"Epoch: {epochs}, accuracy: {accuracy}\n")
    # Repeat until max iterations is reached or we reach a desired accuracy on the validation set
    return neural_network

def predict(testing_set, neural_network):
    # Checking against the validation set
    # For each instance in the validation set:
    tt = 0
    tf = 0
    ft = 0
    ff = 0
    actualLabel = []
    predictLabels = []
    filename = ("results-" + str(file_path) + "-" + str(learning_rate) + "r" + "-" + str(randomSeed) + ".csv")
    for val_i in range(testing_set.shape[0]):

        # 1. Feed instance forward through network
        # A. Calculate out_k for each neuron k in the hidden layer
        out_val_k = []
        # Iterate through all the hidden layer neurons
        for i in range(number_hidden_neurons):
            out_val_k.append(sigmoid(net_calculate(neural_network[0][i], testing_set[val_i])))
        # B. Calculate out_o for each neuron o, using each out_k as the inputs to neuron o
        out_o = sigmoid(net_calculate(neural_network[1], [0] + out_val_k))
        predict = 1 if out_o >= threshold else 0
        predictLabels.append(predict)
        instance_label = testing_set[val_i][0]
        actualLabel.append(instance_label)
        if predict == 1 and instance_label == 1:
            tt += 1
        elif predict == 1 and instance_label == 0:
            ft += 1
        elif predict == 0 and instance_label == 1:
            tf += 1
        else:
            ff += 1
    accuracy = (tt + ff) / (tt + tf + ft + ff)
    print(f"Accuracy: {accuracy}\n")
    #to_confusion_matrix([tt,tf,ft,ff])

    labels = np.unique(actualLabel)
    size = len(actualLabel)
    matrix = dict()

    # create matrix initialised with 0 (nested dictionary)
    for class_name in labels:
        matrix[class_name] = {label: 0 for label in labels}

    # form the confusion matrix by incrementing proper places
    for i in range(size):
        actual_class = actualLabel[i]
        # print("actual_class: ", actual_class)
        pred_class = predictLabels[i]
        # print("pred_class:", pred_class)
        matrix[actual_class][pred_class] += 1
        # print("matrix: ", matrix[actual_class][pred_class])

    matrix = dict(zip(labels, list(matrix.values())))

    print("Confusion Matrix of given model is :")
    print("Predicted Label")
    keys = list(matrix.keys())
    print(",".join(str(e) for e in keys))
    for key, value in matrix.items():
        for pred, count in value.items():
            # print("key, value", key, value)
            print(count, end=",")  # counts in predictLabel & true matching or false counts
        print("%s" % key)  # respective keys
    # print("true, pred: ", true, predictLabel)      # test-related print statement

    with open(filename, "w") as f:
        f.write((",".join(str(e) for e in keys)))
        f.write('\n')
        for key, value in matrix.items():
            for pred, count in value.items():
                f.write(str(count))  # counts in predictLabel & true matching or false counts
                f.write(",")
            f.write("%s" % key)  # respective keys
            f.write("\n")

    return accuracy

"""
Takes the following parameters:

a. The path to a file containing a data set (e.g., monks1.csv)
b. The number of neurons to use in the hidden layer
c. The learning rate 𝜂 to use during backpropagation
d. The percentage of instances to use for a training set
e. A random seed as an integer
f. The threshold to use for deciding if the predicted label is a 0 or 1
"""

# Beginning of code
try:
    # Get Dataset File

    # a.The path to a file containing a data set (e.g., monks1.csv)
    file_path = sys.argv[1]

    # b. The number of neurons to use in the hidden layer
    number_hidden_neurons = int(sys.argv[2])

    # c. The learning rate 𝜂 to use during backpropagation
    learning_rate = float(sys.argv[3])

    # d. The percentage of instances to use for a training set
    training_set_percent = float(sys.argv[4])

    # Ensure training set percent is a valid percent that can be used
    if 0 >= training_set_percent or training_set_percent >= 1:
        print("Invalid percent. Please choose a value between 0 and 1.\n Input can not be 0 or 1 as well")
        exit(1)

    # Store the size of the validation set
    validation_set_percent = (1 - training_set_percent) / 2

    # Store the size of the testing set
    testing_set_percent = (1 - training_set_percent) / 2

    # e. A random seed as an integer
    randomSeed = int(sys.argv[5])

    # f. The threshold to use for deciding if the predicted label is a 0 or 1
    threshold = float(sys.argv[6])

    # e. Number of hidden layers wanted
    if len(sys.argv) > 7:
        hidden_layer_num = int(sys.argv[7])
    else:
        hidden_layer_num = 0

    # Print all input values given for user to see
    print(f"Inputs:\nFile: {file_path}\nLearning rate: {learning_rate}")
    print(
        f"Training Set Percent: {training_set_percent}\nValidation Set and Testing Set Percentages: {validation_set_percent}\n")
    print(f"Random Seed: {randomSeed}\nThreshold: {threshold}")

    # Read in dataset
    df = pd.read_csv(file_path)

    # Shuffle the dataframe. Use random seed from input and fraction 1 as we want the whole dataframe
    shuffled_df = df.sample(frac=1, random_state=randomSeed)

    print(f"Number of Instances in Dataframe: {len(df)}")

    ##TODO: preprocess before splitting
    shuffled_df = data_preprocessing(dataset=shuffled_df).to_numpy()

    # Split Dataset into training, validation, and testing sets. This is through identifying idices of where the percentages are in the dataset
    splits_indices = [int(training_set_percent * len(df)),
                      int((training_set_percent + validation_set_percent) * len(df))]
    print(f"Splits indexes they begin at: {splits_indices}\n")
    training_set, validation_set, testing_set = np.split(shuffled_df, splits_indices)

    # Print out the lengths of the training, validation, and testing sets
    print(f"Length of training: {len(training_set)}")
    print(f"Length of validiation set: {len(validation_set)}")
    print(f"Length of testing: {len(testing_set)}\n")

    # Create all the random beginning weights for each neuron in the hidden layer
    hidden_layer = [np.random.uniform(-0.1, 0.1, training_set.shape[1]) for i in range(number_hidden_neurons)]

    # Create an output neuron that has <number_of_hidden_neurons> TODO double check if correct
    output_neuron = np.random.uniform(-0.1, 0.1, number_hidden_neurons + 1)

    # Create neural network list
    neural_network = [hidden_layer, output_neuron]

    print(training_set.shape)

    # Fit the model using the training set and validation set
    neural_network = fit(training_set, validation_set, neural_network)

    # Output the prediction
    predict(testing_set, neural_network)

except IndexError as e:
    print(f"Error. Message below:\n{e}\nPlease try again.")
    exit(1)
except ValueError as e:
    print(f"Error. Message below:\n{e}\nPlease try again.")
    exit(1)
except FileNotFoundError as e:
    print(f"Error. Message below:\n{e}\nPlease try again.")
    exit(1)

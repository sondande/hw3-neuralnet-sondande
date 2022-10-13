"""
Assignment #3: Neural Network

By Sagana Ondande & Ada Ates
"""

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

"""
Parameters: 
    - Takes in the assumption that the training and validation set are a numpy array
"""

class LogisticalRegressionModel:
    def __init__(self, trainingSet, validationSet, learningRate, threshold_value):
        self.trainSet = trainingSet
        self.valSet = validationSet
        self.learning_rate = learningRate
        self.threshold = threshold_value
        self.weights = np.random.uniform(-0.1, 0.1, trainingSet.shape[1])
        self.test_acc = None

    def fit(self):
        # 2. Until either the accuracy on the validation set > A% or we run n epochs
        # Set accuracy variable
        accuracy = 0
        epochs = 0
        while accuracy <= 0.99:
            # Ensure we are runnning 500 epochs
            if epochs > 500:
                break
            # A. For each instance x in the training set
            for insta_count in range(len(self.trainSet)):
                instance_label = self.trainSet[insta_count][0]
                # Calculate Net Value between X instance and weights
                net = net_calculate(weights=self.weights, x_instance=self.trainSet[insta_count])
                # Calculate the out values from the net values calculated above
                out_value = sigmoid(net=net)
                # I. Calculate gradient of w0
                grad_w0 = 0
                grad_w0 = -1 * out_value * (1 - out_value) * (instance_label - out_value)
                # Update first weight in weights
                self.weights[0] -= (self.learning_rate * grad_w0)
                # II. Calculate gradient of wi
                for attr_count in range(1, len(self.trainSet[0])):
                    grad_wi = 0
                    grad_wi = -1 * self.trainSet[insta_count][attr_count] * out_value * (1 - out_value) * (
                                instance_label - out_value)
                    self.weights[attr_count] -= (self.learning_rate * grad_wi)
            epochs += 1
            tt = 0
            tf = 0
            ft = 0
            ff = 0
            # Testing against validation set
            for insta_count in range(len(self.valSet)):
                instance_label = self.valSet[insta_count][0]
                # Calculate Net Value between X instance and weights
                net = net_calculate(weights=self.weights, x_instance=self.valSet[insta_count])
                # Calculate the out values from the net values calculated above
                out_value = sigmoid(net=net)
                predict = 1 if out_value > self.threshold else 0
                # print(f"Predict Value:{predict}")
                if predict == 1 and instance_label == 1:
                    tt += 1
                elif predict == 1 and instance_label == 0:
                    ft += 1
                elif predict == 0 and instance_label == 1:
                    tf += 1
                else:
                    ff += 1
            accuracy = (tt + ff) / (tt + tf + ft + ff)
        return self.weights # TODO check

    def predict(self, testing_dataset):
        # Initialize Variables
        tt = 0
        tf = 0
        ft = 0
        ff = 0
        accuracy = 0
        actualLabel = []
        predictLabels = []
        for insta_count in range(len(testing_dataset)):
            actualLabel.append(testing_dataset[insta_count][0])
            # Calculate Net Value between X instance and weights
            net = net_calculate(weights=self.weights, x_instance=testing_dataset[insta_count])
            # Calculate the out values from the net values calculated above
            out_value = sigmoid(net=net)
            # Gives a classification to value with the rule: if greater than threshold: 1; else: 0
            predict = 1 if out_value > self.threshold else 0
            predictLabels.append(predict)
            if predict == 1 and testing_dataset[insta_count][0] == 1:
                tt += 1
            elif predict == 1 and testing_dataset[insta_count][0] == 0:
                tf += 1
            elif predict == 0 and testing_dataset[insta_count][0] == 1:
                ft += 1
            else:
                ff += 1
            accuracy = (tt + ff) / (tt + tf + ft + ff)
        # Sets class variable of the test accuracy to the results before returning value
        self.test_acc = accuracy
        return accuracy

    def score(self):
        return self.test_acc

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
            new_dataframe = pd.concat([new_dataframe, dummies],axis=1)
        else: 
            # Find the maximum value in column 'x'
            max_value = max(dataset[x])
            # Find the minimum value in column 'x'
            min_value = min(dataset[x])
            # Check if the column being evaluated is the label column. If so, just add it right into the dataframe
            if x =='label':
                new_dataframe = pd.concat([new_dataframe, dataset[x]], axis=1)
                continue
            # Ensure we don't run into a zero division error when normalizing all the values
            elif (max_value - min_value) != 0:
                # Apply net value formula to every value in pandas dataframe
                dataset[x] = dataset[x].apply(lambda y: (y - min_value)/(max_value - min_value))
                # Combine New column to our new_dataframe
                new_dataframe = pd.concat([new_dataframe, dataset[x]],axis=1)
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
    net = weights[0]
    # Instances and weights are the same length
    for i in range(1, len(x_instance)):
        net += weights[i] * x_instance[i]
    return net

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
    testing_set_percent =  (1 - training_set_percent) / 2

    # e. A random seed as an integer
    randomSeed = int(sys.argv[5])

    #f. The threshold to use for deciding if the predicted label is a 0 or 1
    threshold = float(sys.argv[6])

    # Print all input values given for user to see
    print(f"Inputs:\nFile: {file_path}\nLearning rate: {learning_rate}")
    print(f"Training Set Percent: {training_set_percent}\nValidation Set and Testing Set Percentages: {validation_set_percent}\n")
    print(f"Random Seed: {randomSeed}\nThreshold: {threshold}")

    # Read in dataset
    df = pd.read_csv(file_path)

    # Shuffle the dataframe. Use random seed from input and fraction 1 as we want the whole dataframe
    shuffled_df = df.sample(frac=1,random_state=randomSeed)

    print(f"Number of Instances in Dataframe: {len(df)}")

    # Split Dataset into training, validation, and testing sets. This is through identifying idices of where the percentages are in the dataset
    splits_indices = [int(training_set_percent * len(df)), int((training_set_percent + validation_set_percent) * len(df))]
    print(f"Splits indexes they begin at: {splits_indices}\n")
    training_set, validation_set, testing_set = np.split(shuffled_df, splits_indices)

    # Print out the lengths of the training, validation, and testing sets
    print(f"Length of training: {len(training_set)}")
    print(f"Length of validiation set: {len(validation_set)}")
    print(f"Length of testing: {len(testing_set)}\n")

    # Preprocess the data
    training_set = data_preprocessing(training_set).to_numpy()
    validation_set = data_preprocessing(validation_set).to_numpy()
    testing_set = data_preprocessing(testing_set).to_numpy()

    # Initialize models in hidden layer for the number of hidden layer neurons given as a parameter
    # Notice that at initialization, the weights are different as then created, they are randomly generated in the constructor
    hidden_layer = []
    for x in range(number_hidden_neurons):
        # Creates new logistical regression model
        model = LogisticalRegressionModel(training_set, validation_set, learning_rate, threshold)
        # Trains the model
        model.fit()
        # Calculate predictions from trained model
        model.predict(testing_set)
        # Add trained model to hidden layer list
        hidden_layer.append(model)




    print("hello")
    # Train the model
    # output_weights = model.fit()

    # Calculate prediction
    # model.predict(testing_set)

    # Print score
    # print(f"Score: {model.score()}")

except IndexError as e:
    print(f"Error. Message below:\n{e}\nPlease try again.")
    exit(1)
except ValueError as e:
    print(f"Error. Message below:\n{e}\nPlease try again.")
    exit(1)
except FileNotFoundError as e:
    print(f"Error. Message below:\n{e}\nPlease try again.")
    exit(1)


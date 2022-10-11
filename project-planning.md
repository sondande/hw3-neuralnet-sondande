# Project Planning: 
- Create a class for Logistical Regression: 
  - Parameters would be everything from the prior hw.
    - Training set (same set divided before calling the function)
    - Validation set (same set divided before calling the function)
    - Learning rate 
    - Random seed 
  - Steps: 
    ```
    for length of # of neuron from input: 
    Logical regression(Training set, Validation set, learning rate)
        In the model, it would calculate the weights which would differ because they are randomly generated on execution
    ```
After training: 
Backpropadation

Steps: 
- Loop through each row of the matrix when begining the training and back propagating to compute feedback

Neural Network = [[att], [hidden layer neurons], [output neuron]]

## Asking if we can use: 
- Scipy:
  - sigmoid function

## Function Creation: 
- Score     
  - Calculate the accuracy and produce a confusion matrix on a model with new dataset that isn't the training set
- Signmoid function
  - Scipy
- Back propagation function to update all the weights


# Classes: 
- Logistical Regression Model
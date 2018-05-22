#####################################################################################################################
#	
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#############################################################################################################################


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split;
from sklearn import preprocessing
import sys
from sklearn.model_selection import train_test_split


class NeuralNet:
    def __init__(self, train, header=True, h1=4, h2=2):
        np.random.seed(0)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers

        raw_input = pd.read_csv(train, header=header)
        X = self.preprocess(raw_input)
        ncols = len(X.columns)
        nrows = len(X.index)

        self.X = X.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.y = X.iloc[:, (ncols-1)].values.reshape(nrows, 1)

        # splitting the data into training and test
        self.X, self.X_test, self.y, self.y_test = train_test_split(self.X, self.y)

        #
        # Finding number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        # assigning random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))

    def __activation(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        elif activation == "tanh":
            self.__tanh(self, x)
        elif activation == "relu":
            self.__relu(self, x)

    #
    # Defining the function for tanh, ReLu and their derivatives
    #

    def __activation_derivative(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        elif activation == "tanh":
            self.__tanh_derivative(self, x)
        elif activation == "relu":
            self.__relu_derivative(self, x)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __tanh(self, x):
        return ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))

    def __relu(self, x):
        zeros = np.zeros((x.shape[0],x.shape[1]))
        return np.maximum(zeros, x)

    # derivative of sigmoid function, indicates confidence about existing weight

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __tanh_derivative(self, x):
        return 1 - (x * x)

    def __relu_derivative(self, x):
        x[x > 0] = 1
        x[x < 0] = 0
        return x
    #
    # code for pre-processing the dataset, including standardization, normalization,
    #   categorical to numerical, etc
    #

    def preprocess(self, X):
        X = X.drop_duplicates()
        X = X.replace('\?', np.nan, regex=True)
        X = X.replace('null', np.nan, regex=True)
        X = X.dropna(axis=0, how='any')
        le = preprocessing.LabelEncoder()
        for i in range(X.shape[1]):
            if X[i].dtype == "object":
                X[i] = le.fit_transform(X[i])
        min_max_scaler = preprocessing.MinMaxScaler()
        X = min_max_scaler.fit_transform(X)
        X_normalized = preprocessing.normalize(X, norm='l2')
        X = pd.DataFrame(X_normalized)
        return X

    # Below is the training function

    def train(self, activation, max_iterations, learning_rate):
        for iteration in range(max_iterations):
            out = self.forward_pass(activation)
            error = 0.5 * np.power((out - self.y), 2)
            self.backward_pass(out, activation)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input

        print("After " + str(max_iterations) + " iterations, the total training error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)

    def forward_pass(self,activation):
        if activation == "sigmoid":
            in1 = np.dot(self.X, self.w01)
            self.X12 = self.__sigmoid(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__sigmoid(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__sigmoid(in3)
        elif activation == "tanh":
            in1 = np.dot(self.X, self.w01)
            self.X12 = self.__tanh(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__tanh(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__tanh(in3)
        elif activation == "relu":
            in1 = np.dot(self.X, self.w01)
            self.X12 = self.__relu(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__relu(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__relu(in3)
        return out

    def backward_pass(self, out, activation):
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)

    def compute_output_delta(self, out, activation):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        elif activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))
        elif activation == "relu":
            delta_output = (self.y - out) * (self.__relu_derivative(out))
        self.deltaOut = delta_output

    #Implemented other activation functions

    def compute_hidden_layer2_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
        elif activation == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))
        elif activation == "relu":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__relu_derivative(self.X23))
        self.delta23 = delta_hidden_layer2

    #Implemented other activation functions

    def compute_hidden_layer1_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        elif activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))
        elif activation == "relu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__relu_derivative(self.X12))
        self.delta12 = delta_hidden_layer1

    #Implemented other activation functions

    def compute_input_layer_delta(self, activation):
        if activation == "sigmoid":
            delta_input_layer = np.multiply(self.__sigmoid_derivative(self.X01), self.delta01.dot(self.w01.T))
        elif activation == "tanh":
            delta_input_layer = np.multiply(self.__tanh_derivative(self.X01), self.delta01.dot(self.w01.T))
        elif activation == "relu":
            delta_input_layer = np.multiply(self.__relu_derivative(self.X01), self.delta01.dot(self.w01.T))
        self.delta01 = delta_input_layer

    # Implemented the predict function for applying the trained model on the  test dataset.
    # output - the test error from this function

    def predict(self, activation,header=True):
        self.X = self.X_test
        out = self.forward_pass(activation)
        error = 0.5 * np.power((out - self.y_test), 2)
        return np.sum(error)


if __name__ == "__main__":
      if (len(sys.argv) != 5):
          print("wrong input. please check readme")
           #neural_network = NeuralNet("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
      else:
          trainFilePath = sys.argv[1]
          activation = sys.argv[2]
          maxIterations = int(sys.argv[3])
          learningRate = float(sys.argv[4])
      neural_network = NeuralNet(trainFilePath, None)
      neural_network.train(activation, maxIterations, learningRate)
      testError = neural_network.predict(activation, None)
      print("test error : " + str(testError))
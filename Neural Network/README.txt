Language used -  Python 3.6
Activation functions that can be used - sigmoid/tanh/ReLu

Implemented a Neural Network with an input layer, two hidden layers followed by an output layer

Tested on Datasets - 
Car Evaluation Dataset: https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
Iris Dataset: https://archive.ics.uci.edu/ml/datasets/Iris
Adult Census Income Dataset: https://archive.ics.uci.edu/ml/datasets/
Census+Income

•	Assumptions:
1.	Assumed that last column is the class label column
2.	No bias neurons
3.	Assumed that training and test data have same number of columns
4.	Divided the given datasets into training and testing data (75%,25%).

•	Preprocessing:
1.	Converted all the categorical values into numerical values.
2.	Removed all the duplicate rows.
3.	Removed the rows having cells with“?” value in them.
4.	Removed rows having their cells with NULL values.
5.	Scaled and normalized all the numerical variables.



To Run the code from command line-
python NeuralNet.py datapath activationFunction maxIterations learningRate

import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier

print("Рабочая тетрадь номер 7")

task_list=["1", "2", "3"]

def task1():
    def sigmoid(x):
        return 1 / (1 + numpy.exp(-x))

    class Neuron:
        def __init__(self, weights, bias):
            self.weights = weights
            self.bias = bias

        def feedforward(self, inputs):
            total = numpy.dot(self.weights, inputs) + self.bias
            return sigmoid(total)

    weights = numpy.array([0, 1])
    bias = 4
    n = Neuron(weights, bias)
    x = numpy.array([2, 3])
    print(n.feedforward(x))

    class OurNeuralNetwork:
        def __init__(self):
            weights = numpy.array([0.5, 0.5, 0.5])
            bias = 0
            self.h1 = Neuron(weights, bias)
            self.h2 = Neuron(weights, bias)
            self.h3 = Neuron(weights, bias)
            self.o1 = Neuron(weights, bias)

        def feedforward(self, x):
            out_h1 = self.h1.feedforward(x)
            out_h2 = self.h2.feedforward(x)
            out_h3 = self.h2.feedforward(x)
            out_o1 = self.o1.feedforward(numpy.array([out_h1, out_h2, out_h3]))
            return out_o1

    network = OurNeuralNetwork()
    x = numpy.array([2, 3, 4])
    print(network.feedforward(x))

    class OurNeuralNetwork:
        def __init__(self):
            weights = numpy.array([1, 0])
            bias = 1
            self.h1 = Neuron(weights, bias)
            self.h2 = Neuron(weights, bias)
            self.o1 = Neuron(weights, bias)
            self.o2 = Neuron(weights, bias)

        def feedforward(self, x):
            out_h1 = self.h1.feedforward(x)
            out_h2 = self.h2.feedforward(x)
            out_o1 = self.o1.feedforward(numpy.array([out_h1, out_h2]))
            out_o2 = self.o2.feedforward(numpy.array([out_h1, out_h2]))
            return out_o1, out_o2

    network = OurNeuralNetwork()
    x = numpy.array([2, 3])
    print(network.feedforward(x))

def task2():
    print("Sigmoid")
    def sigmoid(x):
        sig = 1 / (1 + numpy.exp(-x))
        return sig

    class Neuron1:
        def __init__(self, weights, bias):
            self.weights = weights
            self.bias = bias

        def feedforward(self, inputs):
            total = numpy.dot(self.weights, inputs) + self.bias
            return sigmoid(total)

    class OurNeuralNetwork:
        def __init__(self):
            weights = numpy.array([0.5, 0.5, 0.5])
            bias = 0
            self.h1 = Neuron1(weights, bias)
            self.h2 = Neuron1(weights, bias)
            self.h3 = Neuron1(weights, bias)
            self.o1 = Neuron1(weights, bias)

        def feedforward(self, x):
            out_h1 = self.h1.feedforward(x)
            out_h2 = self.h2.feedforward(x)
            out_h3 = self.h3.feedforward(x)
            out_o1 = self.o1.feedforward(numpy.array([out_h1, out_h2, out_h3]))
            return out_o1

    class OrNeuralNetwork:
        def __init__(self):
            weights = numpy.array([1, 0])
            bias = 1

            self.h1 = Neuron1(weights, bias)
            self.h2 = Neuron1(weights, bias)
            self.o1 = Neuron1(weights, bias)
            self.o2 = Neuron1(weights, bias)

        def feedforward(self, x):
            out_h1 = self.h1.feedforward(x)
            out_h2 = self.h2.feedforward(x)
            out_o1 = self.o1.feedforward(numpy.array([out_h1, out_h2]))
            out_o2 = self.o2.feedforward(numpy.array([out_h1, out_h2]))
            return out_o1, out_o2

    network = OurNeuralNetwork()
    x = numpy.array([2, 3, 4])
    print(network.feedforward(x))

    network = OrNeuralNetwork()
    x = numpy.array([2, 3])
    print(network.feedforward(x))

    print("Tanh")
    def tanh(x):
        return numpy.tan(x)

    class Neuron2:
        def __init__(self, weights, bias):
            self.weights = weights
            self.bias = bias

        def feedforward(self, inputs):
            total = numpy.dot(self.weights, inputs) + self.bias
            return tanh(total)

    class OurNeuralNetwork:
        def __init__(self):
            weights = numpy.array([0.5, 0.5, 0.5])
            bias = 0
            self.h1 = Neuron2(weights, bias)
            self.h2 = Neuron2(weights, bias)
            self.h3 = Neuron2(weights, bias)
            self.o1 = Neuron2(weights, bias)

        def feedforward(self, x):
            out_h1 = self.h1.feedforward(x)
            out_h2 = self.h2.feedforward(x)
            out_h3 = self.h3.feedforward(x)
            out_o1 = self.o1.feedforward(numpy.array([out_h1, out_h2, out_h3]))
            return out_o1

    network = OurNeuralNetwork()
    x = numpy.array([2, 3, 4])
    print(network.feedforward(x))

    class OurNeuralNetwork:
        def __init__(self):
            weights = numpy.array([1, 0])
            bias = 1

            self.h1 = Neuron2(weights, bias)
            self.h2 = Neuron2(weights, bias)
            self.o1 = Neuron2(weights, bias)
            self.o2 = Neuron2(weights, bias)

        def feedforward(self, x):
            out_h1 = self.h1.feedforward(x)
            out_h2 = self.h2.feedforward(x)
            out_o1 = self.o1.feedforward(numpy.array([out_h1, out_h2]))
            out_o2 = self.o2.feedforward(numpy.array([out_h1, out_h2]))
            return out_o1, out_o2

    network = OurNeuralNetwork()
    x = numpy.array([2, 3])
    print(network.feedforward(x))

    print("ReLu")
    def ReLU(x):
        return numpy.maximum(0, x)

    class Neuron3:
        def __init__(self, weights, bias):
            self.weights = weights
            self.bias = bias

        def feedforward(self, inputs):
            total = numpy.dot(self.weights, inputs) + self.bias
            return ReLU(total)

    class OurNeuralNetwork:
        def __init__(self):
            weights = numpy.array([0.5, 0.5, 0.5])
            bias = 0
            self.h1 = Neuron3(weights, bias)
            self.h2 = Neuron3(weights, bias)
            self.h3 = Neuron3(weights, bias)
            self.o1 = Neuron3(weights, bias)

        def feedforward(self, x):
            out_h1 = self.h1.feedforward(x)
            out_h2 = self.h2.feedforward(x)
            out_h3 = self.h3.feedforward(x)
            out_o1 = self.o1.feedforward(numpy.array([out_h1, out_h2, out_h3]))
            return out_o1

    network = OurNeuralNetwork()
    x = numpy.array([2, 3, 4])
    print(network.feedforward(x))

    class OurNeuralNetwork:
        def __init__(self):
            weights = numpy.array([1, 0])
            bias = 1

            self.h1 = Neuron3(weights, bias)
            self.h2 = Neuron3(weights, bias)
            self.o1 = Neuron3(weights, bias)
            self.o2 = Neuron3(weights, bias)

        def feedforward(self, x):
            out_h1 = self.h1.feedforward(x)
            out_h2 = self.h2.feedforward(x)
            out_o1 = self.o1.feedforward(numpy.array([out_h1, out_h2]))
            out_o2 = self.o2.feedforward(numpy.array([out_h1, out_h2]))
            return out_o1, out_o2

    network = OurNeuralNetwork()
    x = numpy.array([2, 3])
    print(network.feedforward(x))

def task3():
    url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
    df = pd.read_csv(url)
    df.head(5)
    df
    df = df.rename(columns={'variety': 'target'})
    X_df, Y_df = df.drop(['target'], axis=1), df.target
    print('Dataset Size: ', X_df.shape, Y_df.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(X_df, Y_df, train_size=0.80, test_size=0.20, stratify=Y_df, random_state=123)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    mlp_classifier = MLPClassifier(random_state=123)
    mlp_classifier.fit(X_train, Y_train)
    Y_preds = mlp_classifier.predict(X_test)

    print(Y_preds[:15])
    print(Y_test[:15])
    print('Test Accuracy: %.3f' % mlp_classifier.score(X_test, Y_test))
    print('Training Accuracy: %.3f' % mlp_classifier.score(X_train, Y_train))
    print('Loss: ', mlp_classifier.loss_)
    print('Number of Coefs: ', len(mlp_classifier.coefs_))
    print('Number of Intercepts: ', len(mlp_classifier.intercepts_))
    print('Number of Iteration for Which Estimator Ran: ', mlp_classifier.n_iter_)
    print('Name of Output Layer Activation Function: ', mlp_classifier.out_activation_)

    url = 'https://raw.githubusercontent.com/AnnaShestova/salary-years-simple-linear-regression/master/Salary_Data.csv'
    df = pd.read_csv(url)
    df.head(5)
    df
    df = df.rename(columns={'Salary':'target'})
    X_df, Y_df = df.drop(['target'], axis=1), df.target
    print ('Dataset Size: ', X_df.shape, Y_df.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(X_df, Y_df, train_size = 0.80, test_size = 0.20, random_state = 123)
    print ('Train/Test size: ', X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    mlp_regressor = MLPRegressor(random_state=123)
    mlp_regressor.fit(X_train, Y_train)
    Y_preds = mlp_regressor.predict(X_test)
    print (Y_preds[:10])
    print (Y_test[:10])
    print ('Test R^2 Score: %.3f'%mlp_regressor.score(X_test, Y_test))
    print ('Training R^2 Score: %.3f'%mlp_regressor.score(X_train, Y_train))
    print ('Loss: ', mlp_regressor.loss_)
    print ('Number of Coefs: ', len(mlp_regressor.coefs_))
    print ('Number of Intercepts: ', len(mlp_regressor.intercepts_))
    print ('Number of Iteration for Which Estimator Ran: ', mlp_regressor.n_iter_)
    print ('Name of Output Layer Activation Function: ', mlp_regressor.out_activation_)

def main():
    print()
    print("Список заданий:")
    print(task_list)
    current_task=str(input("Введите номер задания: "))
    if current_task=="0":
        exit()
    if current_task=="1":
        task1()
    if current_task=="2":
        task2()
    if current_task=="3":
        task3()

while True:
    main()
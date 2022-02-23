# import numpy as np


def sigmoid(z):
    return 1.0/(1.0 + pow(2.71828, -z))


def dot_product(x, w):
    return sum([i*j for (i, j) in zip(x, w)])


def feedforward(x, w, th):
    import numpy as np
    a = sigmoid(np.dot(w, x) + th)
    return a


def predict(x, w, th):
    a = feedforward(x, w, th)
    if(a>0.5):
        return 1
    return 0


#reading data and spliting it to train and test set
train_x = [[0, 0], [0, 1], [1, 0], [1, 1]]
test_x = [[0, 0], [0, 1], [1, 0], [1, 1]]

train_y = [0, 0, 0, 1]
test_y = [0, 0, 0, 1]


# initializing weight vector
import random
input_size = len(train_x[0])
w1 = random.uniform(-0.5, 0.5)
w2 = random.uniform(-0.5, 0.5)
w = [[w1, w2]]
th = 0.5 # thershold value
epochs = 100
a = 0.1 # learning rate 


def learn(X, Y, w, th, a, epochs):
    for t in range(epochs):
        for i in range(len(X)):
            x, y = X[i], Y[i]
            y_pred = feedforward(x, w, th)
            w = w + a*(y - y_pred)*X[i]
            th = th + a*(y - y_pred)   
            # print('weight\t', w, ' thershold' ,th,'epoch', t)
    return w, th



# using learn to train the perceptron on training dataset
w, th = learn(train_x, train_y, w, th, a, epochs)
w_and = w

# Using perceptron to predict for test dataset
print('AND GATE')
print('Desired Result\t', 'Predicted Result')
for i in range(len(test_x)):
    x = test_x[i] 
    y = test_y[i]
    y_pred = predict(x, w, th)
    print(y, '\t\t\t', y_pred)


train_x = [[0, 0], [0, 1], [1, 0], [1, 1]]
test_x = [[0, 0], [0, 1], [1, 0], [1, 1]]

train_y = [0, 1, 1, 1]
test_y = [0, 1, 1, 1]

# using learn to train the perceptron on training dataset
w, th = learn(train_x, train_y, w, th, a, epochs)
w_or = w

# Using perceptron to predict for test dataset
print('OR GATE')
print('Desired Result\t', 'Predicted Result')
for i in range(len(test_x)):
    x = test_x[i] 
    y = test_y[i]
    y_pred = predict(x, w, th)
    print(y, '\t\t\t', y_pred)


train_x = [[0, 0], [0, 1], [1, 0], [1, 1]]
test_x = [[0, 0], [0, 1], [1, 0], [1, 1]]

train_y = [1, 1, 1, 0]
test_y = [1, 1, 1, 0]

# using learn to train the perceptron on training dataset
w, th = learn(train_x, train_y, w, th, a, epochs)
w_nand = w

# Using perceptron to predict for test dataset
print('NAND GATE')
print('Desired Result\t', 'Predicted Result')
for i in range(len(test_x)):
    x = test_x[i] 
    y = test_y[i]
    y_pred = predict(x, w, th)
    print(y, '\t\t\t', y_pred)


train_x = [[0, 0], [0, 1], [1, 0], [1, 1]]
test_x = [[0, 0], [0, 1], [1, 0], [1, 1]]

train_y = [1, 0, 0, 0]
test_y = [1, 0, 0, 0]

# using learn to train the perceptron on training dataset
w, th = learn(train_x, train_y, w, th, a, epochs)
w_nor = w

# Using perceptron to predict for test dataset
print('NOR GATE')
print('Desired Result\t', 'Predicted Result')
for i in range(len(test_x)):
    x = test_x[i] 
    y = test_y[i]
    y_pred = predict(x, w, th)
    print(y, '\t\t\t', y_pred)


train_x = [[0, 0], [0, 1], [1, 0], [1, 1]]
test_x = [[0, 0], [0, 1], [1, 0], [1, 1]]

train_y = [0, 1, 1, 0]
test_y = [0, 1, 1, 0]

# using learn to train the perceptron on training dataset
wand, th = learn(train_x, [0, 0, 0, 1], w, th, a, epochs)
wor, th = learn(train_x, [0, 1, 1, 1], w, th, a, epochs)
wnand, th = learn(train_x, [1, 1, 1, 0], w, th, a, epochs)
# w, th = learn(train_x, train_y, w, th, a, epochs)
w_xor = w

# Using perceptron to predict for test dataset
print('XOR GATE')
print('Desired Result\t', 'Predicted Result')
for i in range(len(test_x)):
    x = test_x[i] 
    y = test_y[i]
    y_pred1 = predict(x, wor, th)
    y_pred2 = predict(x, wnand, th)
    y_pred = predict([y_pred1, y_pred2], wand, th)
    print(y, '\t\t\t', y_pred)

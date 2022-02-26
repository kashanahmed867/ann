import csv
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1.0/(1.0+pow(2.718, -z))


def feedforward(x, w, th):
    a = sigmoid((w @ x) + th)
    return a


def predict(x, w, th):
    a = feedforward(x, w, th)
    if(a>0.5):
        return 1
    return 0


def assign_value_to_classes(arr, cls, tgt):
    import numpy as np
    updated_list = []
    for i in range(len(arr)):
        z = tgt[cls.index(arr[i])]
        updated_list.append(z)
    return np.array(updated_list)


#reading data and spliting it to train and test set
import numpy as np
import pandas as pd
dataSet = pd.read_csv('dataset.csv') #reading and shuffling data 
selected = np.random.rand(len(dataSet)) < 0.70

train = dataSet[selected]
test = dataSet[~selected]

train_y = train['variety'].values
test_y = test['variety'].values

train_y = assign_value_to_classes(train_y, ['Setosa', 'Versicolor'], [0, 1])
test_y = assign_value_to_classes(test_y, ['Setosa', 'Versicolor'], [0, 1])

train_x = train.drop(columns = ['variety']).values
test_x = test.drop(columns = ['variety']).values

points = dataSet.drop(columns = ['variety']).values


# initializing weight vector
input_size = len(train_x[0])
w = np.zeros((1, input_size))
th = 0                                   # thershold value
epochs = 20
a = 0.1                                   # learning rate 


def learn(X, Y, w, th, a, epochs):
    for t in range(epochs):
        for i in range(len(X)):
            x, y = X[i], Y[i]
            y_pred = feedforward(x, w, th)
            w = w + a*(y - y_pred)*X[i]
            th = th + a*(y - y_pred)   
            #print('weight\t', w, ' thershold' ,th,'epoch', t)
    return w, th


def plot(train_x, w, th):
        wt = w[0].tolist()
        """Ploting scatterplot of data"""
        plt.scatter(train_x[:50, 0], train_x[:50, 1], color = 'red', marker = 'o', 
            label = 'Setosa')
        plt.scatter(train_x[50:100, 0], train_x[50:100, 1], color = 'blue', marker = 'x', 
            label = 'Versicolor')
        plt.xlabel('Petal Length')
        plt.ylabel('Sepal Length')
        plt.legend(loc='lower right')

        """
        plotting line
        to plot the line we need to calculate it's slope and y-intercept
        the formula described below gives us what we need:
        x2 = -(w1/w2)x1-(b/w2)
        """
        slope = -1*(wt[0]/wt[1])
        intercept = -1*(th[0]/wt[1])        
        axes = plt.gca()
        print(axes)
        x_vals = np.array(axes.get_xlim())
        print(x_vals)
        y_vals = intercept + slope * x_vals
        print(y_vals)
        plt.plot(x_vals, y_vals, '-')
        plt.show()


# using learn to train the perceptron on training dataset
w, th = learn(train_x, train_y, w, th, a, epochs)

# Using perceptron to predict for test dataset
print('True label\t', 'predicted label')
for i in range(len(test_x)):
    x = test_x[i] 
    y = test_y[i]
    y_pred = predict(x, w, th)
    print(y, '\t\t\t', y_pred)

# plotting graph here
plot(points, w, th)

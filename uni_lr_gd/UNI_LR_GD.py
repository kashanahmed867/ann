import matplotlib.pyplot as plt


# calculating cost for objective function
def calculate_cost(thetas, x, y, m):
    summation = 0
    for i in range(m):
        summation += ((thetas[0] + thetas[1]*x[i] - y[i]) ** 2)
    cost = summation / (2*m)
    return cost

# calculating derivate with respect to theta 0
def deriv_theta_0(thetas, x, y, m):
    summation = 0
    for i in range(m):
        summation += thetas[0] + thetas[1]*x[i] - y[i]
    summation = summation / m
    return summation

# calculating derivative with respect to theta 1
def deriv_theta_1(thetas, x, y, m):
    summation = 0
    for i in range(m):
        summation += ((thetas[0] + thetas[1]*x[i] - y[i]) * x[i])
    summation = summation / m
    return summation

# calculating first-ordered derivate with respect to theta 0 second-ordered derivate with respect to theta 0
def deriv_theta_00(thetas, x, y, m):
    summation = 0
    for i in range(m):
        summation += 1
    summation = summation / m
    return summation

# calculating first-ordered derivate with respect to theta 0 second-ordered derivate with respect to theta 1
def deriv_theta_01(thetas, x, y, m):
    summation = 0
    for i in range(m):
        summation += x[i]
    summation = summation / m
    return summation

# calculating first-ordered derivate with respect to theta 1 second-ordered derivate with respect to theta 0
def deriv_theta_10(thetas, x, y, m):
    summation = 0
    for i in range(m):
        summation += x[i]
    summation = summation / m
    return summation

# calculating first-ordered derivate with respect to theta 1 second-ordered derivate with respect to theta 1
def deriv_theta_11(thetas, x, y, m):
    summation = 0
    for i in range(m):
        summation += x[i]**2
    summation = summation / m
    return summation
# ------------------------------------------------------------------


# check saddle points of cost function using hessian method and update thetas on these points
def handle_saddle_points(thetas, x, y, m):
    f00 = deriv_theta_00(thetas, x, y, m)
    f01 = deriv_theta_01(thetas, x, y, m)
    f10 = deriv_theta_10(thetas, x, y, m)
    f11 = deriv_theta_11(thetas, x, y, m)

    hessian = (f00*f11) - (f01*f10)

    # there is saddle point when hessian is negative
    if hessian < 0:
        #change in thetas to escape from saddle points
        thetas[0] = thetas[0] + 0.001
        thetas[1] = thetas[1] + 0.001
    return thetas

# training of perceptron using batch gradient descent approach, also update theta values in every pass
def batch_gradient_descent(iterations, alpha, train_data, train_data_labels, total_train_samples, momentum, thetas):
    # calculate initial cost of function
    prev_cost = cost = calculate_cost(thetas, train_data, train_data_labels, total_train_samples)
    # set change for momentum
    prev_change_0 = prev_change_1 = 0.0
    # prepare data storage for history
    history_theta_0 = [thetas[0]]
    history_theta_1 = [thetas[1]]
    history_cost = [cost]
    i = 0
    while(cost >= 0 and i <= iterations and cost <= prev_cost):
        # there are a lot of methods to excape from the saddle points like hessian method, SGD method, optimization methods like momentum, adam etc.
        # below line is commented as we are already using momentum method to take care of saddle points
        # thetas = handle_saddle_points(thetas, train_data, train_data_labels, total_train_samples)
        change_0 = alpha*(deriv_theta_0(thetas, train_data, train_data_labels, total_train_samples)) + (momentum * prev_change_0)
        change_1 = alpha*(deriv_theta_1(thetas, train_data, train_data_labels, total_train_samples)) + (momentum * prev_change_1)
        thetas[0] = thetas[0] - change_0
        thetas[1] = thetas[1] - change_1
        # save current change of thetas and current change of cost to use for next iteration
        prev_change_0 = change_0
        prev_change_1 = change_1
        prev_cost = cost
        # calculate cost of function
        cost = calculate_cost(thetas, train_data, train_data_labels, total_train_samples)
        # store thetas and cost in history
        history_theta_0.append(thetas[0])
        history_theta_1.append(thetas[1])
        history_cost.append(cost)
        i+=1
    return thetas, history_theta_0, history_theta_1, history_cost
# ------------------------------------------------------------------


def save_predictions(thetas, x, y, m):
    # writing predictions for testing dateset in Predictions.csv file
    predictions = open("Predictions.csv", "w")
    predictions.write("X-Value,Actual-Value,Predicted-Value,Least-Square-Error\n")
    total_error = 0
    for i in range(m):
        y_predicted = thetas[0] + thetas[1]*x[i]
        least_square_error = (y_predicted - y[i]) ** 2
        predictions.write("{xa},{ya},{yp},{lse}\n".format(xa=x[i], ya=y[i], yp=y_predicted, lse=least_square_error))
        total_error += least_square_error
    average_error = total_error / m
    predictions.write("Average Error: {ar}".format(ar=average_error))
    predictions.close()

def save_cost_theta_0(alpha, history_theta_0, history_cost):
    # writing theta0 and thier cost values in CostFunction_Theta0.csv file
    theta_0_costs_file = open("CostFunction_Theta0.csv", "w")
    theta_0_costs_file.write("Theta0_Value,CostFunction\n")
    for i in range(len(history_theta_0)):
        theta_0_costs_file.write("{t0},{c}\n".format(t0=history_theta_0[i], c=history_cost[i]))
    theta_0_costs_file.write("Learning Rate: {lr}".format(lr=alpha))
    theta_0_costs_file.close()

def save_cost_theta_1(alpha, history_theta_1, history_cost):
    # writing theta1 and thier cost values in CostFunction_Theta1.csv file
    theta_1_costs_file = open("CostFunction_Theta1.csv", "w")
    theta_1_costs_file.write("Theta1_Value,CostFunction\n")
    for i in range(len(history_theta_1)):
        theta_1_costs_file.write("{t1},{c}\n".format(t1=history_theta_1[i], c=history_cost[i]))
    theta_1_costs_file.write("Learning Rate: {lr}".format(lr=alpha))
    theta_1_costs_file.close()
# ------------------------------------------------------------------


# learning rate or step parameter
alpha = 0.02
# initializing thetas with 0
thetas = [0, 0]
# maximum number of iterations
iterations = 1000
# define momentum
momentum = 0.001
# list of training features and labels
train_data = []
train_data_labels = []
# list of testing features and labels
test_data = []
test_data_labels = []
# ------------------------------------------------------------------


# taking inputs from user
data_filename = input("Enter the Name of Train Data File: ")
alpha = input("Enter the Value of Learning Rate: ")
alpha = float(alpha)
test_filename = input("Enter the Name of Test Data: ")


# open dataset file
data_file = open(data_filename, "r")
test_file = open(test_filename, "r")

for x in data_file:
    line = x.split(",")
    # converting string to float
    train_data.append(float(line[0]))
    train_data_labels.append(float(line[1]))

for x in test_file:
    line = x.split(",")
    # converting string to float
    test_data.append(float(line[0]))
    test_data_labels.append(float(line[1]))
# ------------------------------------------------------------------


total_train_samples = len(train_data)
total_test_samples = len(test_data)
print("Total number of training samples in given dataset are " + str(total_train_samples))
print("Total number of testing samples in given dataset are " + str(total_test_samples))

# using batch gradient descent method with momentum to handle saddle points
trained_thetas, history_theta_0, history_theta_1, history_cost = batch_gradient_descent(iterations, alpha, train_data, train_data_labels, total_train_samples, momentum, thetas)
print("Final Thetas using perceptron based gradient descent method are " + str(trained_thetas))

# storing output results in csv files
save_predictions(trained_thetas, test_data, test_data_labels, total_test_samples)
save_cost_theta_0(alpha, history_theta_0, history_cost)
save_cost_theta_1(alpha, history_theta_1, history_cost)
# ------------------------------------------------------------------


m = trained_thetas[1]
c = trained_thetas[0]
predicted_data = [(m*x)+c for x in train_data]

plt.figure(figsize=(10,8))
# adjusting ranges and interval of x and y axis
plt.xticks([i for i in range(4,25,2)])
plt.yticks([i for i in range(-5,26,5)])
# scatter all training data point in graph
plt.scatter(train_data, train_data_labels, marker = "o", c = "red", label = 'Training Data')
# labelling the x and y axis
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
# plot and show graph
plt.plot(train_data, predicted_data, label = 'Regression Line')
plt.legend(loc='best')
plt.show()
# ------------------------------------------------------------------


# use numpy and axes3d only for 3d plotting as there is no way to handle without them
import numpy as np
from mpl_toolkits.mplot3d import axes3d

theta_range = 10
cost_values = np.zeros((theta_range, theta_range))
for i in range(1, theta_range+1):
    for j in range(1, theta_range+1):
        cost_values[i-1, j-1] = calculate_cost([i, j], train_data, train_data_labels, total_train_samples)

fig = plt.figure(figsize = (10, 8))
ax = fig.gca(projection = '3d')
xy_range = [x for x in range(1, theta_range+1)]
X, Y = np.meshgrid(xy_range, xy_range)
surf = ax.plot_surface(X, Y, cost_values, cmap = 'viridis', rstride = 1, cstride = 1)
fig.colorbar(surf, shrink = 0.5, aspect = 5)
# set labels & view angle and show 3d graph
plt.xlabel("$\Theta_0$")
plt.ylabel("$\Theta_1$")
ax.set_zlabel("$J(\Theta)$")
ax.view_init(30, 330) #initial angle
plt.show()
# ------------------------------------------------------------------


fig, ax1 = plt.subplots()
# plot thetas over time
color='tab:blue'
ax1.plot(history_theta_0, label='$\\theta_{0}$', linestyle='--', color=color)
ax1.plot(history_theta_1, label='$\\theta_{1}$', linestyle='-', color=color)
ax1.set_xlabel('Iterations'); ax1.set_ylabel('$\\theta$', color=color);
ax1.tick_params(axis='y', labelcolor=color)
# plot cost function over time
color='tab:red'
ax2 = ax1.twinx()
ax2.plot(history_cost, label='Cost Function', color=color)
ax2.set_title('Values of $\\theta$ and $J(\\theta)$ over iterations')
ax2.set_ylabel('Cost: $J(\\theta)$', color=color)
ax1.tick_params(axis='y', labelcolor=color)
fig.legend()
plt.show()
# ------------------------------------------------------------------

# 2d graph between thetas 0 and cost
plt.figure(figsize=(10,8))
plt.gca()
plt.plot(history_theta_0, history_cost)
plt.xlabel("$\Theta_0$")
plt.ylabel("$J(\Theta)$")
plt.show()

# 2d graph between thetas 1 and cost
plt.figure(figsize=(10,8))
plt.gca()
plt.plot(history_theta_1, history_cost)
plt.xlabel("$\Theta_1$")
plt.ylabel("$J(\Theta)$")
plt.show()
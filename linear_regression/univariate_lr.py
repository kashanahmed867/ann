import matplotlib.pyplot as plt


# calculating cost
def calculate_cost(thetas, x, y, m):
    summation = 0
    for i in range(total_samples):
        summation += ((thetas[0] + thetas[1]*x[i] - y[i]) ** 2)
    cost = summation / (2*m)
    return cost

# compare two model theta values and select the best one with minimum cost
def compare_models(thetas_prev, thetas_current, x, y, m):
    cost_prev = calculate_cost(thetas_prev, x, y, m)
    cost_current = calculate_cost(thetas_current, x, y, m)
    min_cost = (cost_prev, cost_current) [cost_current < cost_prev]
    best_thetas = (thetas_prev, thetas_current) [cost_current < cost_prev]
    return min_cost, best_thetas

# run every possible combination of theta values within given range
def best_model_in_range(x1_range, x0_range, x, y, m, thetas):
    # initial setting of values
    last_iteration = thetas
    last_min_cost = 999999999 # setting max possible cost
    last_best_thetas = [1, 1]
    for i in range(1, x1_range+1):
        for j in range(1, x0_range+1):
            min_cost, best_thetas = compare_models(last_iteration, [i, j], x, y, m)
            last_min_cost = (last_min_cost, min_cost) [min_cost < last_min_cost]
            last_best_thetas = (last_best_thetas, best_thetas) [min_cost < last_min_cost]
            last_iteration = [i, j]
    return last_min_cost, last_best_thetas
# ------------------------------------------------------------------


# find best fit line directly by mathematical calculations in one go
def find_best_fit_line(x, y):
    sum_x = sum(x)
    sum_y = sum(y)
    mean_x = sum_x / len(x)
    mean_y = sum_y / len(y)
    m_dividend = sum([((m-mean_x)*(n-mean_y)) for m, n in zip(x, y)])
    m_divisor = sum([((m-mean_x)**2) for m in x])
    m_slope = m_dividend/m_divisor
    b_intercept = mean_y - (m_slope * mean_x)
    return [b_intercept, m_slope]
# ------------------------------------------------------------------


# calculating R-squared value for measuring goodness of our model. 
def calculate_r2(thetas, x, y, m):
    mean_y = sum(y) / len(y)
    ss_t = 0 #total sum of squares
    ss_r = 0 #total sum of square of residuals
    for i in range(m): # val_count represents the no.of input x values
        y_pred = thetas[0] + thetas[1] * x[i]
        ss_t += (y[i] - mean_y) ** 2
        ss_r += (y[i] - y_pred) ** 2
        r2 = 1 - (ss_r/ss_t)
    return r2

# compare two model theta values and select the best one with maximum R-squared value
def compare_models_r2(thetas_prev, thetas_current, x, y, m):
    r2_prev = calculate_r2(thetas_prev, x, y, m)
    r2_current = calculate_r2(thetas_current, x, y, m)
    max_r2 = (r2_prev, r2_current) [r2_current > r2_prev]
    best_thetas = (thetas_prev, thetas_current) [r2_current > r2_prev]
    return max_r2, best_thetas

# run every possible combination of theta values within given range and find maximum R-squared value
def best_model_in_range_r2(x1_range, x0_range, x, y, m, thetas):
    # initial setting of values
    last_iteration = thetas
    last_max_r2 = -999999999 # setting min possible r2
    last_best_thetas = [1, 1]
    for i in range(1, x1_range+1):
        for j in range(1, x0_range+1):
            max_r2, best_thetas = compare_models_r2(last_iteration, [i, j], x, y, m)
            last_max_r2 = (last_max_r2, max_r2) [max_r2 > last_max_r2]
            last_best_thetas = (last_best_thetas, best_thetas) [max_r2 > last_max_r2]
            last_iteration = [i, j]
    return last_max_r2, last_best_thetas
# ------------------------------------------------------------------


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

# training of perceptron using batch gradient descent approach, also update theta values in every pass
def batch_gradient_descent(iterations, alpha, train_data, train_data_labels, total_samples, thetas):
    cost = calculate_cost(thetas, train_data, train_data_labels, total_samples)
    i = 0
    while(cost >= 0 and i <= iterations):
        thetas[0] = thetas[0] - alpha*(deriv_theta_0(thetas, train_data, train_data_labels, total_samples))
        thetas[1] = thetas[1] - alpha*(deriv_theta_1(thetas, train_data, train_data_labels, total_samples))
        cost = calculate_cost(thetas, train_data, train_data_labels, total_samples)
        i+=1
    return thetas
# ------------------------------------------------------------------


# parameters
alpha = 0.02
# initializing thetas with 0
thetas = [0, 0]
# maximum number of iterations
iterations = 1000
# list of training features
train_data = []
# list of training labels
train_data_labels = []
# ------------------------------------------------------------------


# open dataset file
file = open("data.csv", "r")

for x in file:
    line = x.split(",")
    # converting string to float
    train_data.append(float(line[0]))
    train_data_labels.append(float(line[1]))

total_samples = len(train_data)
print("Total number of samples in given dataset are " + str(total_samples))

theta_range = int(input("Enter range of theta for finding minimum cost from 1 to n: "))
# ------------------------------------------------------------------


# method 1: using range constraint and calculating minimum cost
range_cost, range_thetas = best_model_in_range(theta_range, theta_range, train_data, train_data_labels, total_samples, thetas)
print("Method 1 Results: Final Thetas using different theta values in given range are " + str(range_thetas) + " with minimum cost of " + str(round(range_cost, 3)))

# method 2: using range constraint and calculating maximum R-squared value
range_r2, range_thetas_r2 = best_model_in_range_r2(theta_range, theta_range, train_data, train_data_labels, total_samples, thetas)
print("Method 2 Results: Final Thetas using different theta values in given range are " + str(range_thetas_r2) + " with maximum r2 value of " + str(round(range_r2, 3)))

# method 3: using perceptron training with gradient descent approach
trained_thetas = batch_gradient_descent(iterations, alpha, train_data, train_data_labels, total_samples, thetas)
print("Method 3 Results: Final Thetas using perceptron based gradient descent method are " + str(trained_thetas))

# method 4: using finding best fit line directly
best_fit_thetas = find_best_fit_line(train_data, train_data_labels)
print("Method 4 Results: Final Thetas using best fit line method are " + str(best_fit_thetas))
# ------------------------------------------------------------------


m = best_fit_thetas[1]
c = best_fit_thetas[0]
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

cost_values = np.zeros((theta_range, theta_range))
for i in range(1, theta_range+1):
    for j in range(1, theta_range+1):
        cost_values[i-1, j-1] = calculate_cost([i, j], train_data, train_data_labels, total_samples)

fig = plt.figure(figsize = (10, 8))
ax = fig.gca(projection = '3d')
xy_range = [x for x in range(1, theta_range+1)]
surf = ax.plot_surface(xy_range, xy_range, cost_values, cmap = 'viridis')
fig.colorbar(surf, shrink = 0.5, aspect = 5)
# set labels & view angle and show 3d graph
plt.xlabel("$\Theta_0$")
plt.ylabel("$\Theta_1$")
ax.set_zlabel("$J(\Theta)$")
ax.view_init(30, 330) #initial angle
plt.show()

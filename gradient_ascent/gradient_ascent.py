import matplotlib.pyplot as plt
from matplotlib import cm


# objective function
def objective(x, y):
	return x**2.0 - y**2.0
 
# derivative of objective function
def gradient_vector(vector):
	return [derivative_x(vector[0]), derivative_y(vector[1])]

# partial derivative w.r.t x
def derivative_x(x):
	return x * 2.0

# partial derivative w.r.t y
def derivative_y(y):
	return -(y * 2.0)

# using linear space without using numpy
def linspace(start, stop, steps):
    step = (stop - start) / steps
    return [start+(x*step) for x in range(steps+1)]
# ------------------------------------------------------------------


# gradient ascent algorithm
def gradient_ascent(iterations, step_size, starting_point):
	# track all points
	points, points_x, points_y, scores = list(), list(), list(), list()
	# generate an initial point
	point = starting_point
	# run the gradient ascent
	for i in range(iterations):
		# calculate gradient
		gradient = gradient_vector(point)
		# take a step
		point[0] = point[0] + step_size * gradient[0]
		point[1] = point[1] + step_size * gradient[1]
		# evaluate candidate point
		score = objective(point[0], point[1])
		# store point
		points_x.append(point[0])
		points_y.append(point[1])
		points.append(point)
		scores.append(score)

	return [points, points_x, points_y, scores]
# ------------------------------------------------------------------


# parameters
alpha = 0.05
# maximum number of iterations
iterations = 20
# initial point
starting_point = [1, 1]

points, points_x, points_y, scores = gradient_ascent(iterations, alpha, starting_point)
# ------------------------------------------------------------------


import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d') # Create the axes

# Data in linear space
X = np.linspace(-10, 10, 100)
Y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(X, Y)
Z = objective(X, Y)

# Plot the 3d surface
surface = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rstride = 2, cstride = 2)

plt.plot(points_x, points_y, scores, '.-', color='red')

# Set some labels
ax.set_xlabel('B0')
ax.set_ylabel('B1')
ax.set_zlabel('fn')

plt.show()

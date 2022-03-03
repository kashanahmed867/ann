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
def gradient_ascent(iterations, step_size, starting_point, boundary_limit):
	# track all points
	points, points_x, points_y, scores = list(), list(), list(), list()
	# generate an initial point
	point = starting_point
	# set minimum possible score for loop termination if upcoming score is decreasing
	last_score = -boundary_limit
	# run the gradient ascent
	for i in range(iterations):
		# calculate gradient
		gradient = gradient_vector(point)
		# take a step
		point[0] = point[0] + step_size * gradient[0]
		point[1] = point[1] + step_size * gradient[1]
		# evaluate candidate point
		score = objective(point[0], point[1])
		# terminate iterations if score exceeding graph boundary limit or score decreasing as per requirement
		if score >= boundary_limit or last_score >= score: break
		# store point
		last_score = score # for conditional termination
		points_x.append(point[0])
		points_y.append(point[1])
		points.append(point)
		scores.append(score)

	return [points, points_x, points_y, scores]
# ------------------------------------------------------------------


# learning rate or step, step 2 is very large so taking small step
alpha = 0.2
# maximum number of iterations
iterations = 20
# boundary limit for final score
boundary_limit = 100
# initial point
starting_point = [1, 1]
# try random starting point
# import random
# starting_point = [random.randint(-10,10), random.randint(-10,10)]

# getting result of gradient ascent
points, points_x, points_y, scores = gradient_ascent(iterations, alpha, starting_point, boundary_limit)
# ------------------------------------------------------------------


import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d') # Create the axes

# Data in linear space
X = np.linspace(-10, 10, 100)
Y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(X, Y)
Z = objective(X, Y) # For final score

# Plot the 3d surface
surface = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rstride = 2, cstride = 2)

plt.plot(points_x, points_y, scores, '.-', color='red')

# Set some labels
ax.set_xlabel('B0')
ax.set_ylabel('B1')
ax.set_zlabel('Score')

plt.show()
#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt


###################################### Define initial functions ##########################################

# This solves the diffusion equation portion of the ODE we want to solve
def solve_diffusion(beta, Ngrid, sigma):
    A = np.eye(Ngrid) + (1.0 + 2.0 * beta)+ np.eye(Ngrid, k=1) * -beta + np.eye(Ngrid, k=-1) * -beta
    sigma = np.linalg.solve(A, sigma)
    return sigma
# Recall that sigma starts as a sharp function
def get_sigma_initial(x):
    mean = 1.
    std = 0.05
    return np.exp(np.divide(np.multiply(-1., np.power(np.subtract(x, mean), 2.)), (2 * np.power(std, 2.))))


################################ Set up parameters and initial conditions ###############################
dx = 0.01                                        # x step as seen in class
Ngrid = int(1./dx)                               # Grid size as seen in class
Nsteps = 1000                                    # Number of steps as seen in class
nu = 0.01                                        # nu is the kinematic viscosity
D = 3*nu                                         # D is the diffusion coefficient
x = np.multiply(np.arange(0, 1., dx), 2.)        # x is the radial distance x = r/R_0

u = np.divide(-3. * D / 2., x[1:])               # u is the velocity
u = np.append([0], u)                            # Set the appropriate boundary conditions
u[0] = -np.abs(u[1])
u[-1] = np.abs(u[-2])

dt = np.min(np.abs(np.divide(dx, u)))            # We use the Courant condition to get dt
alpha = np.multiply(u, dt / (2. * dx))           # alpha and beta as seen in class
beta = D * dt / (dx ** 2.)
sigma = get_sigma_initial(x)                     # Get the starting sigma value

############################################### Solve and plot ##############################################

# Set up graph, title, labels, limits, etc
plt.ion()
fig, ax = plt.subplots(1, 1)
ax.plot(x, sigma, 'k-')
plot, = ax.plot(x, sigma, 'ro')
ax.set_title('Evolution of Surface Density')
ax.set_xlabel('Radial distance x = r/R_0')
ax.set_ylabel('Surface Density')
ax.legend(['Initial Surface Density', 'Surface Density'])
fig.canvas.draw()
ax.set_xlim([0., 2.])
ax.set_ylim([0., 1.])



for ct in range(Nsteps):

    # Solve for the diffusion part of the equation
    solve_diffusion(beta, Ngrid, sigma)

    # Advection evolution using the Lax-Friedrich method
    sigma[1:(Ngrid - 1)] = 0.5 * (sigma[2:] + sigma[:(Ngrid - 2)]) - alpha[1:(Ngrid - 1)] * (sigma[2:] - sigma[:(Ngrid - 2)])

    # Plot and evolve the data
    sigma[0] = sigma[1]
    sigma[-1] = sigma[-2]
    plot.set_ydata(sigma)
    fig.canvas.draw()
    plt.pause(0.0001)


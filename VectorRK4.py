#A program to implement Vector RK4 Method to solve a system of 1st order ODEs and graph output

#Import necessary libraries
from numpy import * #Necessary for sine/cosine/pi etc
import matplotlib.pyplot as plt #Necessary for graphing

#Parameters go here
step = 0.01
k = pi**2 #Spring constant (for the system)
m = 1 #Mass (for the system)
t_initial = 0 #Starting values for the solution
x_initial = 1 #Starting position (for the system)
v_initial = 0 #Starting velocity (for the system)
u_initial = [x_initial, v_initial]
t_final = 8

#Define the f(t,u) vector in du/dt=f(t,u)
def f(t, u):
    x = u[0]
    v = u[1]
    return [v, -k * x / m]

def RK4MethodStep(t, u):
    k1 = f(t, u)
    k2 = f(t + step/2, add(u, multiply(step/2, k1)))
    k3 = f(t + step/2, add(u, multiply(step/2, k2)))
    k4 = f(t + step, add(u, multiply(step, k3)))
    return add(u, multiply(step/6, add(add(add(k1, multiply(2, k2)), multiply(2, k3)), k4)))

#Initialize lists to store the values of the points
t_values = [t_initial]
u_values = [u_initial]
while t_values[-1] < t_final: #Loop until you reach or pass the end point
    u_values.append(RK4MethodStep(t_values[-1], u_values[-1])) #Add the next u values (using RK4 Method)
    t_values.append(t_values[-1] + step) #Add the next t value
    
#Extract the data from the u values list of lists
x_values = transpose(u_values)[0]
v_values = transpose(u_values)[1]

#Graph the output
plt.figure()
plt.plot(t_values, x_values, 'bo')
plt.grid(True)
plt.show()
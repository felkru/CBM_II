import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# Constants
m1 = 1
m2 = 1
h = 0.1

# Initial values for the state vector: [x1, y1, x2, y2, px1, py1, px2, py2]
StateVectorEuler = np.array([0, 0, 10, 0, 0, -0.2, 0, 0.2])
StateVectorBetterEuler = np.array(StateVectorEuler)
StateVectorRK45 = np.array(StateVectorEuler)  # For RK45

# Define the system of ODEs
def TheFunction(t, theState):
    x1, y1, x2, y2, px1, py1, px2, py2 = theState
    x1d = px1 / m1
    y1d = py1 / m1
    x2d = px2 / m2
    y2d = py2 / m2
    common = m1 * m2 * ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (-3 / 2)
    p1xd = (x2 - x1) * common
    p1yd = (y2 - y1) * common
    p2xd = (x1 - x2) * common
    p2yd = (y1 - y2) * common
    return np.array([x1d, y1d, x2d, y2d, p1xd, p1yd, p2xd, p2yd])

# Euler integration
def SimpleEuler(theState):
    return theState + h * TheFunction(0, theState)

def BetterEuler(theState):
    g = theState + h / 2 * TheFunction(0, theState)
    return theState + h * TheFunction(0, g)

# RK45 integration step
def RK45Step(theState, t, h):
    sol = solve_ivp(TheFunction, (t, t + h), theState, method='RK45', t_eval=[t + h])
    return sol.y[:, -1]  # Return the last state after the step

fig, ax = plt.subplots()
scat = ax.scatter(
    [StateVectorEuler[0], StateVectorEuler[2], StateVectorBetterEuler[0], StateVectorBetterEuler[2], StateVectorRK45[0], StateVectorRK45[2]],
    [StateVectorEuler[1], StateVectorEuler[3], StateVectorBetterEuler[1], StateVectorBetterEuler[3], StateVectorRK45[1], StateVectorRK45[3]],
    c=['blue', 'blue', 'red', 'red', 'green', 'green']  # Blue: SimpleEuler, Red: BetterEuler, Green: RK45
)

ax.set(xlim=[-20, 20], ylim=[-20, 20])

t = 0  # Initial time

def update(frame):
    global StateVectorEuler, StateVectorBetterEuler, StateVectorRK45, t
    StateVectorEuler = SimpleEuler(StateVectorEuler)
    StateVectorBetterEuler = BetterEuler(StateVectorBetterEuler)
    StateVectorRK45 = RK45Step(StateVectorRK45, t, h)
    t += h  # Increment time
    scat.set_offsets([
        [StateVectorEuler[0], StateVectorEuler[1]], [StateVectorEuler[2], StateVectorEuler[3]],
        [StateVectorBetterEuler[0], StateVectorBetterEuler[1]], [StateVectorBetterEuler[2], StateVectorBetterEuler[3]],
        [StateVectorRK45[0], StateVectorRK45[1]], [StateVectorRK45[2], StateVectorRK45[3]]
    ])
    return scat,

ani = animation.FuncAnimation(fig, update, frames=200, interval=1, blit=False)

plt.show()
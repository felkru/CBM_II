
import numpy as np
import matplotlib.pyplot as plt

# --- Aufgabe 2: Adaptive Schrittweiten ---

# This part assumes a pre-existing solution for the 3-body problem.
# For demonstration purposes, I will use a simplified ODE system.
# In a real scenario, the user would provide their 3-body problem solution.

# Simplified ODE system for demonstration (e.g., dy/dt = -y)
def simplified_ode(t, y):
    return -y

# Euler step function
def euler_step(y, t, h, ode_func):
    return y + h * ode_func(t, y)

def AdaptiveEulerStep_HalfStepMethod(y, t, h, epsilon, hmax=1):
    # Calculate one Euler step with full step size h
    y_full_step = euler_step(y, t, h, simplified_ode)

    # Calculate two Euler steps with half step size h/2
    y_half_step_1 = euler_step(y, t, h/2, simplified_ode)
    y_half_step_2 = euler_step(y_half_step_1, t + h/2, h/2, simplified_ode)

    # Calculate local error e (2-norm of the difference)
    e = np.linalg.norm(y_full_step - y_half_step_2)

    # Determine new step size h_new
    if e == 0:
        h_new = 2 * h
    else:
        h_new = h * np.min([np.max([epsilon / e, 0.1]), 5])

    # If error is within tolerance, return solution with new step size
    if e <= epsilon:
        return y_half_step_2, t + h, h_new
    else:
        # If error is too large, repeat calculation with h/2
        return AdaptiveEulerStep_HalfStepMethod(y, t, h/2, epsilon, hmax)

# Example usage (simplified for demonstration)
initial_y = np.array([1.0])
initial_t = 0.0
initial_h = 0.1
epsilon = 1e-3
hmax = 1.0

# Store results for plotting
t_values = [initial_t]
y_values = [initial_y[0]]
h_values = [initial_h]

current_y = initial_y
current_t = initial_t
current_h = initial_h

# Simulate for a few steps
for _ in range(20):
    next_y, next_t, next_h = AdaptiveEulerStep_HalfStepMethod(current_y, current_t, current_h, epsilon, hmax)
    t_values.append(next_t)
    y_values.append(next_y[0])
    h_values.append(next_h)

    current_y = next_y
    current_t = next_t
    current_h = next_h

    if current_t > 5: # Stop condition for simulation
        break

# Plotting results
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t_values, y_values, label='Adaptive Euler Solution')
plt.xlabel('Time')
plt.ylabel('y')
plt.title('Adaptive Euler Method (Simplified ODE)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t_values, h_values, label='Adaptive Step Size (h)', color='orange')
plt.xlabel('Time')
plt.ylabel('Step Size (h)')
plt.title('Adaptive Step Size over Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('adaptive_euler_results.png')
plt.show()


# d) Was beobachten Sie im Vergleich zu den anderen Algorithmen?
# e) Wie verhält sich die adaptive Schrittweite?
# These will be answered in the analysis document.



#!/usr/bin/env python
# coding: utf-8

# # Basic SZR model

# In[74]:


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function that returns dS/dt, dZ/dt, dR/dt
def model(y, tau, B, N, delta):
    S, Z, R = y
    dSdt = -B * S * Z / N
    dZdt = B * S * Z / N - delta * Z
    dRdt = delta * Z
    return [dSdt, dZdt, dRdt]

# Initial conditions
S0 = 900   # Initial susceptible population
Z0 = 100   # Initial infected (zombie) population
R0 = 0      # Initial removed (dead or immune) population
y0 = [S0, Z0, R0]

# Parameters
B = 0.1  # Rate of zombie-human encounters leading to infection
N = 1000  # Total population size
delta = 0.005 # Rate at which zombies decay

# Time points
tau = np.linspace(0, 1000, 100)

# Solve ODE
sol = odeint(model, y0, tau, args=(B, N, delta))
S, Z, R = sol.T

# Plot
plt.figure(figsize=(10, 6))
plt.plot(tau, S, label='Susceptible')
plt.plot(tau, Z, label='Infected (Zombie)')
plt.plot(tau, R, label='Removed')
plt.xlabel('Dimensionless Time (tau)')
plt.ylabel('Population')
plt.title('Zombie Apocalypse Simulation')
plt.legend()
plt.grid(True)
plt.show()


# In[43]:


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function that returns dS/dt, dZ/dt, dR/dt
def model(y, tau, B, N, delta):
    S, Z, R = y
    dSdt = -B * S * Z / N
    dZdt = B * S * Z / N - delta * Z
    dRdt = delta * Z
    return [dSdt, dZdt, dRdt]

# Initial conditions
S0 = 900   # Initial susceptible population
Z0 = 100   # Initial infected (zombie) population
R0 = 0      # Initial removed (dead or immune) population
y0 = [S0, Z0, R0]

# Parameters
B = 0.03  # Rate of zombie-human encounters leading to infection
N = 1000  # Total population size
delta = 0.005 # Rate at which zombies are removed (killed)

# Time points
tau = np.linspace(0, 2000, 1000)

# Solve ODE
sol = odeint(model, y0, tau, args=(B, N, delta))
S, Z, R = sol.T

# Plot
plt.figure(figsize=(10, 6))
plt.plot(tau, S, label='Susceptible')
plt.plot(tau, Z, label='Infected (Zombie)')
plt.plot(tau, R, label='Removed')
plt.xlabel('Dimensionless Time (tau)')
plt.ylabel('Population')
plt.title('Zombie Apocalypse Simulation')
plt.legend()
plt.grid(True)
plt.show()


# In[51]:


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function that returns dS/dt, dZ/dt, dR/dt
def model(y, tau, B, N, delta):
    S, Z, R = y
    dSdt = -B * S * Z / N
    dZdt = B * S * Z / N - delta * Z
    dRdt = delta * Z
    return [dSdt, dZdt, dRdt]

# Initial conditions
S0 = 9000   # Initial susceptible population
Z0 = 1  # Initial infected (zombie) population
R0 = 0      # Initial removed (dead or immune) population
y0 = [S0, Z0, R0]

# Parameters
B = 0.1  # Rate of zombie-human encounters leading to infection
N = 9001  # Total population size
delta = 0.005 # Rate at which zombies are removed (killed)

# Time points
tau = np.linspace(0, 2000, 1000)

# Solve ODE
sol = odeint(model, y0, tau, args=(B, N, delta))
S, Z, R = sol.T

# Plot
plt.figure(figsize=(10, 6))
plt.plot(tau, S, label='Susceptible')
plt.plot(tau, Z, label='Infected (Zombie)')
plt.plot(tau, R, label='Removed')
plt.xlabel('Dimensionless Time (tau)')
plt.ylabel('Population')
plt.title('Zombie Apocalypse Simulation')
plt.legend()
plt.grid(True)
plt.show()


# ## DFE

# In[15]:


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function that returns dS/dt, dZ/dt, dR/dt
def model(y, tau, B, N, delta):
    S, Z, R = y
    dSdt = -B * S * Z / N
    dZdt = B * S * Z / N - delta * Z
    dRdt = delta * Z
    return [dSdt, dZdt, dRdt]

# Initial conditions
S0 = 1000    # Initial susceptible population
Z0 = 0     # Initial infected (zombie) population
R0 = 0      # Initial removed (dead or immune) population
y0 = [S0, Z0, R0]

# Parameters
B = 0.1  # Rate of zombie-human encounters leading to infection
N = 1000    # Total population size
delta = 0.005 # Rate at which zombies are removed (killed)

# Time points
tau = np.linspace(0, 700, 100)

# Solve ODE
sol = odeint(model, y0, tau, args=(B, N, delta))
S, Z, R = sol.T

# Plot
plt.figure(figsize=(10, 6))
plt.plot(tau, S, label='Susceptible')
plt.plot(tau, R, label='Removed')
plt.xlabel('Dimensionless Time (tau)')
plt.ylabel('Population')
plt.title('Zombie Apocalypse Simulation')
plt.legend()
plt.grid(True)
plt.show()


# # Phase Plane for basic SZR

# In[38]:


import numpy as np
import matplotlib.pyplot as plt

# Create grid
U, V = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))

# Define the differential equations for the SIR model
def sir_model(u, v, R0):
    du_dt = -R0 * u * v
    dv_dt = (R0 * u - 1) * v
    return du_dt, dv_dt

# Additional constraint line
constraint_line = np.linspace(0, 1, 100)
constraint_u = constraint_line
constraint_v = 1 - constraint_line

# Values of R0
R0_values = [0.5, 2, 3]

# Plotting phase planes
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for i, R0 in enumerate(R0_values):
    # Compute derivatives at each grid point
    du_dt, dv_dt = sir_model(U, V, R0)
    # Mask velocities above the line by setting them to zero
    mask = U + V > 0.99
    du_dt[mask] = 0
    dv_dt[mask] = 0
    # Plot phase plane with streamlines
    strm = axs[i].streamplot(U, V, du_dt, dv_dt, density=0.6, color='black')
    # Add constraint line
    axs[i].plot(constraint_u, constraint_v, linestyle='--', color='green')  # Nullcline is dashed
    if R0 > 1:
        # Calculate intersection point of the nullcline with the simplex boundary
        nullcline_intersection_u = 1 / (R0)  # Calculate intersection point u-coordinate
        nullcline_intersection_v = 1 - nullcline_intersection_u  # Calculate intersection point v-coordinate
        nullcline_x = [0.5, nullcline_intersection_u]  # x-coordinates of the nullcline intersection line
        nullcline_y = [0, nullcline_intersection_v]  # y-coordinates of the nullcline intersection line
        axs[i].plot(nullcline_x, nullcline_y, color='red', label='Nullcline Intersection Line')  # Plot nullcline intersection line
    axs[i].set_xlabel('susceptible fraction')
    axs[i].set_ylabel('zombie fraction')
    axs[i].set_xlim(0, 1)
    axs[i].set_ylim(0, 1)
    axs[i].legend()
    axs[i].set_title(f'SZR epidemic, $R_0$={R0}')

plt.tight_layout()
plt.show()


# In[53]:


import numpy as np
import matplotlib.pyplot as plt

# Create grid
U, V = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))

# Define the differential equations for the SIR model
def sir_model(u, v, R0):
    du_dt = -R0 * u * v
    dv_dt = (R0 * u - 1) * v
    return du_dt, dv_dt

# Additional constraint line
constraint_line = np.linspace(0, 1, 100)
constraint_u = constraint_line
constraint_v = 1 - constraint_line

# Values of R0
R0_values = [0.5, 2, 3]

# Plotting phase planes
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for i, R0 in enumerate(R0_values):
    # Compute derivatives at each grid point
    du_dt, dv_dt = sir_model(U, V, R0)
    # Mask velocities above the line by setting them to zero
    mask = U + V > 0.99
    du_dt[mask] = 0
    dv_dt[mask] = 0
    # Plot phase plane with streamlines
    strm = axs[i].streamplot(U, V, du_dt, dv_dt, density=0.6, color='black')
    
    # Add directional field arrows
    axs[i].quiver(U, V, du_dt, dv_dt, scale=5, color='blue')

    # Add constraint line
    axs[i].plot(constraint_u, constraint_v, linestyle='--', color='green')  # Nullcline is dashed
    if R0 > 1:
        # Calculate intersection point of the nullcline with the simplex boundary
        nullcline_intersection_u = 1 / (R0)  # Calculate intersection point u-coordinate
        nullcline_intersection_v = 1 - nullcline_intersection_u  # Calculate intersection point v-coordinate
        nullcline_x = [0.5, nullcline_intersection_u]  # x-coordinates of the nullcline intersection line
        nullcline_y = [0, nullcline_intersection_v]  # y-coordinates of the nullcline intersection line
        axs[i].plot(nullcline_x, nullcline_y, color='red', label='Nullcline Intersection Line')  # Plot nullcline intersection line
    axs[i].set_xlabel('susceptible fraction')
    axs[i].set_ylabel('zombie fraction')
    axs[i].set_xlim(0, 1)
    axs[i].set_ylim(0, 1)
    axs[i].legend()
    axs[i].set_title(f'SZR epidemic, $R_0$={R0}')

plt.tight_layout()
plt.show()


# ## zombsim with probabilities
# 

# In[4]:


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function that returns dS/dt, dZ/dt, dR/dt
def model(y, t, beta, alpha, sigma, N, gamma):
    S, Z, R = y
    dSdt = - beta * S * Z / N - sigma * S * Z / N
    dZdt = beta * S * Z / N - alpha * S * Z / N - gamma * Z 
    dRdt = sigma * S * Z / N + alpha * S * Z / N + gamma * Z 
    return [dSdt, dZdt, dRdt]

# Initial conditions
S0 = 900   # Initial susceptible population
Z0 = 100   # Initial infected (zombie) population
R0 = 0     # Initial removed (dead or immune) population
y0 = [S0, Z0, R0]

# Parameters
N = 1000   # Total population size
zeta = 2  # Zombie encounter rate

# Define probabilities
P1 = 0.3   # Zombie attacking human
P2 = 0.03  # Human attacking zombie
P3 = 0.15   # Outright death by zombie

# Calculate rates
beta = P1 * zeta
alpha = P2 * zeta
sigma = P3 * zeta
gamma = 1/30

# Time points
t = np.linspace(0, 50, 100000)

# Solve ODE
sol = odeint(model, y0, t, args=(beta, alpha, sigma, N, gamma))
S, Z, R = sol.T

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, Z, label='Infected (Zombie)')
plt.plot(t, R, label='Removed')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SZR Model Simulation')
plt.legend()
plt.grid(True)
plt.show()


# # zombsim Preston zeta = 20

# In[66]:


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function that returns dS/dt, dZ/dt, dR/dt
def model(y, t, beta, alpha, sigma, N, gamma):
    S, Z, R = y
    dSdt = - beta * S * Z / N - sigma * S * Z / N
    dZdt = beta * S * Z / N - alpha * S * Z / N - gamma * Z 
    dRdt = sigma * S * Z / N + alpha * S * Z / N + gamma * Z 
    return [dSdt, dZdt, dRdt]

# Initial conditions
S0 = 147800   # Initial susceptible population
Z0 = 1   # Initial infected (zombie) population
R0 = 0     # Initial removed (dead or immune) population
y0 = [S0, Z0, R0]

# Parameters
N = 147801   # Total population size
zeta = 20  # Zombie encounter rate

# Define probabilities
P1 = 0.25   # Zombie infecting human
P2 = 0.01  # Human attacking zombie
P3 = 0.2   # Outright death by zombie

# Calculate rates
beta = P1 * zeta
alpha = P2 * zeta
sigma = P3 * zeta
gamma = 1/30 # zombie decay after 30 days

# Time points
t = np.linspace(0, 30, 100)

# Solve ODE
sol = odeint(model, y0, t, args=(beta, alpha, sigma, N, gamma))
S, Z, R = sol.T

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, Z, label='Infected (Zombie)')
plt.plot(t, R, label='Removed')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SZR Model Simulation')
plt.legend()
plt.grid(True)
plt.show()


# ## infection = attack rate

# In[31]:


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function that returns dS/dt, dZ/dt, dR/dt
def model(y, t, beta, alpha, sigma, N, gamma):
    S, Z, R = y
    dSdt = - beta * S * Z / N - sigma * S * Z / N
    dZdt = beta * S * Z / N - alpha * S * Z / N - gamma * Z 
    dRdt = sigma * S * Z / N + alpha * S * Z / N + gamma * Z 
    return [dSdt, dZdt, dRdt]

# Initial conditions
S0 = 147800   # Initial susceptible population
Z0 = 10000   # Initial infected (zombie) population
R0 = 0     # Initial removed (dead or immune) population
y0 = [S0, Z0, R0]

# Parameters
N = S0 + Z0 + R0   # Total population size
zeta = 20  # Zombie encounter rate

# Define probabilities
P1 = 0.05   # Zombie infecting human
P2 = 0.05  # Human attacking zombie
P3 = 0.05   # Outright death by zombie

# Calculate rates
beta = P1 * zeta
alpha = P2 * zeta
sigma = P3 * zeta
gamma = 1/30 # zombie decay after 30 days

# Time points
t = np.linspace(0, 300, 100)

# Solve ODE
sol = odeint(model, y0, t, args=(beta, alpha, sigma, N, gamma))
S, Z, R = sol.T

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, Z, label='Infected (Zombie)')
plt.plot(t, R, label='Removed')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SZR Model Simulation')
plt.legend()
plt.grid(True)
plt.show()


# # SZFR modelling
# ## B2>B

# In[107]:


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function that returns dS/dt, dZ/dt, dR/dt
def model(y, tau, B, N, O, delta, B2, O2, alpha, alpha2):
    S, Z, R, F = y
    dSdt = -B * S * Z / N -O*S*Z/N
    dZdt = B * S * Z / N - delta * Z + B2 *F*Z/N - alpha *S*Z/N - alpha2 *F*Z/N
    dRdt = delta * Z + (alpha *S*Z/N) +(O *S*Z/N) + (O2 *F*Z/N) + (alpha2 *F*Z/N)
    dFdt = -B2 *F*Z/N - O2 *F*Z/N
    return [dSdt, dZdt, dRdt, dFdt]

# Initial conditions
S0 = 8500   # Initial susceptible population
Z0 = 500  # Initial infected (zombie) population
R0 = 0 # Initial removed (dead or immune) population
F0 = 1000 # initial fighters
y0 = [S0, Z0, R0, F0]

# Parameters
B = 0.25  # Rate of zombie-human encounters leading to infection
N = 10000  # Total population size
delta = 0.005 # Rate at which zombies decay
B2 = 0.35 # zombies infecting fighters
O2 = 0.2 #zombie outright kill fighter
O = 0.1 #zombie outright kill susceptible
alpha = 0.025 # susceptible killing zombie
alpha2 = 0.15 #fighter killing zombie

# Time points
tau = np.linspace(0, 200, 100)

# Solve ODE
sol = odeint(model, y0, tau, args=(B, N, delta, O, O2, B2, alpha, alpha2))
S, Z, R, F = sol.T

# Plot
plt.figure(figsize=(10, 6))
plt.plot(tau, S, label='Susceptible')
plt.plot(tau, Z, label='Infected (Zombie)')
plt.plot(tau, R, label='Removed')
plt.plot(tau, F, label='Fighters')
plt.xlabel('Dimensionless Time (tau)')
plt.ylabel('Population')
plt.title('Zombie Apocalypse Simulation with fighters')
plt.legend()
plt.grid(True)
plt.show()


# ## B2<B

# In[109]:


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function that returns dS/dt, dZ/dt, dR/dt
def model(y, tau, B, N, O, delta, B2, O2, alpha, alpha2):
    S, Z, R, F = y
    dSdt = -B * S * Z / N -O*S*Z/N
    dZdt = B * S * Z / N - delta * Z + B2 *F*Z/N - alpha *S*Z/N - alpha2 *F*Z/N
    dRdt = delta * Z + alpha *S*Z/N + O *S*Z/N + O2 *F*Z/N + alpha2 *F*Z/N
    dFdt = -B2 *F*Z/N - O2 *F*Z/N
    return [dSdt, dZdt, dRdt, dFdt]

# Initial conditions
S0 = 8500   # Initial susceptible population
Z0 = 500  # Initial infected (zombie) population
R0 = 0 # Initial removed (dead or immune) population
F0 = 1000 # initial fighters
y0 = [S0, Z0, R0, F0]

# Parameters
B = 0.35  # Rate of zombie-human encounters leading to infection
N = S0 + Z0 + F0 + R0  # Total population size
delta = 0.005 # Rate at which zombies decay
B2 = 0.25 # zombies infecting fighters
O2 = 0.2 #zombie outright kill fighter
O = 0.1 #zombie outright kill susceptible
alpha = 0.025 # susceptible killing zombie
alpha2 = 0.15 #fighter killing zombie

# Time points
tau = np.linspace(0, 200, 100)
# Solve ODE
sol = odeint(model, y0, tau, args=(B, N, delta, O, O2, B2, alpha, alpha2))
S, Z, R, F = sol.T

# Plot
plt.figure(figsize=(10, 6))
plt.plot(tau, S, label='Susceptible')
plt.plot(tau, Z, label='Infected (Zombie)')
plt.plot(tau, R, label='Removed')
plt.plot(tau, F, label='Fighters')
plt.xlabel('Dimensionless Time (tau)')
plt.ylabel('Population')
plt.title('Zombie Apocalypse Simulation with fighters')
plt.legend()
plt.grid(True)
plt.show()


# In[27]:


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function that returns dS/dt, dZ/dt, dR/dt
def model(y, tau, B, N, O, delta, B2, O2, alpha, alpha2):
    S, Z, R, F = y
    dSdt = -B * S * Z / N -O*S*Z/N
    dZdt = B * S * Z / N - delta * Z + B2 *F*Z/N - alpha *S*Z/N - alpha2 *F*Z/N
    dRdt = delta * Z + alpha *S*Z/N + O *S*Z/N + O2 *F*Z/N + alpha2 *F*Z/N
    dFdt = -B2 *F*Z/N - O2 *F*Z/N
    return [dSdt, dZdt, dRdt, dFdt]

# Initial conditions
S0 = 128000   # Initial susceptible population
Z0 = 100  # Initial infected (zombie) population
R0 = 0 # Initial removed (dead or immune) population
F0 = 20000 # initial fighters
y0 = [S0, Z0, R0, F0]

# Parameters
B = 0.15  # Rate of zombie-human encounters leading to infection
N = S0 + Z0 + F0 + R0  # Total population size
delta = 0.005 # Rate at which zombies decay
B2 = 0.02 # zombies infecting fighters
O2 = 0.05 #zombie outright kill fighter
O = 0.05 #zombie outright kill susceptible
alpha = 0.025 # susceptible killing zombie
alpha2 = 0.2 #fighter killing zombie

# Time points
tau = np.linspace(0, 500, 100)
# Solve ODE
sol = odeint(model, y0, tau, args=(B, N, delta, O, O2, B2, alpha, alpha2))
S, Z, R, F = sol.T

# Plot
plt.figure(figsize=(10, 6))
plt.plot(tau, S, label='Susceptible')
plt.plot(tau, Z, label='Infected (Zombie)')
plt.plot(tau, R, label='Removed')
plt.plot(tau, F, label='Fighters')
plt.xlabel('Dimensionless Time (tau)')
plt.ylabel('Population')
plt.title('Zombie Apocalypse Simulation with fighters')
plt.legend()
plt.grid(True)
plt.show()


# # SIZR 

# In[85]:


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function that returns dS/dt, dZ/dt, dR/dt
def model(y, tau, B, N, rho, delta):
    S, I, Z, R = y
    dSdt = -B * S * Z / N
    dIdt = B*S*Z/N - rho*I
    dZdt = rho*I - delta * Z
    dRdt = delta * Z

    return [dSdt, dIdt, dZdt, dRdt]

# Initial conditions
S0= 1000   # Initial susceptible population
I0= 0 # initial infectious
Z0 = 1  # Initial infected (zombie) population
R0 = 0 # Initial removed (dead or immune) population

y0 = [S0, I0, Z0, R0]

# Parameters
B = 0.2  # Rate of zombie-human encounters leading to infection
N = S0 + I0 + Z0 + R0  # Total population size
delta = 0.3 # Rate at which zombies decay
rho = 0.04 # rate infectious travel to z class

# Time points
tau = np.linspace(0, 500, 100)

# Solve ODE
sol = odeint(model, y0, tau, args=(B, N, delta, rho))
S, I, Z, R = sol.T

# Plot
plt.figure(figsize=(10, 6))
plt.plot(tau, S, label='Susceptible')
plt.plot(tau, Z, label='Infected (Zombie)')
plt.plot(tau, R, label='Removed')
plt.plot(tau, I, label='Infectious')
plt.xlabel('Dimensionless Time (tau)')
plt.ylabel('Population')
plt.title('Zombie Apocalypse Simulation with Infectious')
plt.legend()
plt.grid(True)
plt.show()


# In[94]:


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function that returns dS/dt, dZ/dt, dR/dt
def model(y, tau, B, N, rho, delta):
    S, I, Z, R = y
    dSdt = -B * S * Z / N
    dIdt = B*S*Z/N - rho*I
    dZdt = rho*I - delta * Z
    dRdt = delta * Z

    return [dSdt, dIdt, dZdt, dRdt]

# Initial conditions
S0 = 8982000   # Initial susceptible population
I0 = 0      # initial infectious
Z0 = 100000      # Initial infected (zombie) population
R0 = 0      # Initial removed (dead or immune) population

y0 = [S0, I0, Z0, R0]

# Parameters
B = 0.4  # Rate of zombie-human encounters leading to infection
N = S0 + I0 + Z0 + R0  # Total population size
delta = 0.3 # Rate at which zombies decay
rho = 1/5.6 # Rate at which infectious individuals transition to zombie class

# Time points
tau = np.linspace(0, 200, 100)

# Solve ODE
sol = odeint(model, y0, tau, args=(B, N, delta, rho))
S, I, Z, R = sol.T

# Plot
plt.figure(figsize=(10, 6))
plt.plot(tau, S, label='Susceptible')
plt.plot(tau, Z, label='Infected (Zombie)')
plt.plot(tau, R, label='Removed')
plt.plot(tau, I, label='Infectious')
plt.xlabel('Dimensionless Time (tau)')
plt.ylabel('Population')
plt.title('Zombie Apocalypse Simulation with Infectious (London)')
plt.legend()
plt.grid(True)
plt.show()


# In[100]:


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function that returns dS/dt, dZ/dt, dR/dt
def model(y, tau, B, N, rho, delta, O, O2, B2, A, A2, O3):
    S, F, I, Z, R = y
    dSdt = -B * S * Z / N -O*S*Z/N
    dFdt = -B2*F*Z/N - O2*F*Z/N
    dIdt = B*S*Z/N - rho*I + B2*F*Z/N -O3*I*Z/N
    dZdt = rho*I - delta * Z +B2*F*Z/N +B*S*Z/N  -A*S*Z/N -A2*F*Z/N
    dRdt = delta * Z +A*S*Z/N + A2*F*Z/N + O*S*Z/N + O2*F*Z/N +O3*I*Z/N

    return [dSdt, dFdt, dIdt, dZdt, dRdt]

# Initial conditions
S0 = 10000   # Initial susceptible population
F0 = 1000
I0 = 0# initial infectious
Z0 = 100      # Initial infected (zombie) population
R0 = 0      # Initial removed (dead or immune) population

y0 = [S0, F0, I0, Z0, R0]

# Parameters
B = 0.4  # Rate of zombie-human encounters leading to infection
N = S0 + F0 + I0 + Z0 + R0  # Total population size
delta = 0.3 # Rate at which zombies decay
rho = 1/5.6 # Rate at which infectious individuals transition to zombie class
O = 0.02
O2 = 0.04
O3 = 0.02
A = 0.02
A2 = 0.04
B2 = 0.03


# Time points
tau = np.linspace(0, 200, 100)

# Solve ODE
sol = odeint(model, y0, tau, args=(B, N, rho, delta, O, O2, B2, A, A2, O3))
S, F, I, Z, R = sol.T

# Plot
plt.figure(figsize=(10, 6))
plt.plot(tau, S, label='Susceptible')
plt.plot(tau, Z, label='Infected (Zombie)')
plt.plot(tau, R, label='Removed')
plt.plot(tau, I, label='Infectious')
plt.plot(tau, F, label='fighters')
plt.xlabel('Dimensionless Time (tau)')
plt.ylabel('Population')
plt.title('Zombie Apocalypse Simulation with Infectious (London)')
plt.legend()
plt.grid(True)
plt.show()


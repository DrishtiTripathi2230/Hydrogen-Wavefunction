import matplotlib.pyplot as plt
import numpy as np
from scipy.special import sph_harm, genlaguerre #sph_harm for angular wave function spherical harmonic and genlaguerre for laguerre polynomial
a0 = 1 #keeping 1 in atomic units is standard practice in quantum mechanics simulations.

def radial_wavefunction(r,n,l):
  #Radial part distance from nucleus :
  rho = 2*r/(n*a0)
  normalization = np.sqrt((2/(n*a0))**3*np.math.factorial(n-l-1)/(2*n*np.math.factorial(n+1)))
  Lag = genlaguerre(n-l-1,2*l+1)(rho)
  return normalization * np.exp(-rho/2)*rho**l*Lag

def angular_wavefunction(theta, phi, l, m):
  return sph_harm(m,l,phi,theta)  #sperical harnomic for finding angular wavefunction
# Function to generate wavefunction for any (n, l, m)

def hydrogen_wavefunction(n, l, m, grid_size=50):
    # Create 3D spherical coordinate grid
    r = np.linspace(0, 10, grid_size)
    theta = np.linspace(0, np.pi, grid_size)
    phi = np.linspace(0, 2 * np.pi, grid_size)
    r, theta, phi = np.meshgrid(r, theta, phi, indexing="ij")
    
    R = radial_wavefunction(r,n,l)
    Y = angular_wavefunction(theta, phi, l, m)
    psi = R*Y  #wavefunction
    probability_density = np.abs(psi)**2 #probabilty density

    # Convert to Cartesian for plotting
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x,y,z,probability_density

n, l, m = 4, 3, 2
# Generate wavefunction data
x, y, z, probability_density = hydrogen_wavefunction(n, l, m)

# Plot 3D Electron Density
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with color mapping
ax.scatter(x, y, z, c=probability_density, cmap='plasma', alpha=0.1)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(f"Hydrogen Orbital (n={n}, l={l}, m={m})")

plt.show()


import numpy as np
import matplotlib.pyplot as plt

# --- Parámetros ---
E = 1
mu = 1
g = 1 

# --- Semiejes ---
rmax = np.sqrt(2 * (E + mu*g))
drmax = np.sqrt(2 * (E + mu*g) / (mu + 1))

# --- Malla ---
x = np.linspace(-rmax, rmax, 500)
dx = np.linspace(-rmax, drmax, 500)
x, dx = np.meshgrid(x, dx)

# --- Funciones de las regiones ---
Z1 = (x**2 + dx**2) / (2*(E + mu*g))
Z2 = (x**2 + (mu + 1)*dx**2) / (2*(E + mu*g))

# --- Unión de las regiones ---
region1 = (dx < 0) & (Z1 <= 1)
region2 = (dx >= 0) & (Z2 < 1)
Z = (region1 | region2).astype(float)

# --- Gráfica ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(x, dx, Z, levels=[0.5, 1.5], colors=['#a8dadc'], alpha=0.7)

# --- Fronteras analíticas ---
t1 = np.linspace(np.pi, 2*np.pi, 500)
t2 = np.linspace(0, np.pi, 500)

# Semicircunferencia inferior
ax.plot(rmax*np.cos(t1), rmax*np.sin(t1), color='#1d3557', linewidth=2)

# Semielipse superior
ax.plot(rmax*np.cos(t2), drmax*np.sin(t2), color='#1d3557', linewidth=2, linestyle='--')

# --- Etiquetas ---
ax.set_xlabel(r'$X$', fontsize=16)
ax.set_ylabel(r'$\dot{X}$', fontsize=16)
ax.tick_params(axis='both', labelsize=12)
ax.text(0.97, 0.97, rf'$\mu = {mu}$', transform=ax.transAxes,
        ha='right', va='top', fontsize=12)
ax.text(0.03, 0.97, rf'$E = {E}$', transform=ax.transAxes,
        ha='left', va='top', fontsize=12)
ax.text(0.97, 0.03, rf'$\gamma = {g}$', transform=ax.transAxes,
        ha='right', va='bottom', fontsize=12)
ax.set_xlim(-rmax, rmax)
ax.set_ylim(-rmax, drmax)
ax.set_aspect('equal')
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

fig.savefig("re.png", format='png', bbox_inches='tight', dpi=800)
plt.show()

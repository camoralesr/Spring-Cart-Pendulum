import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# -----------------------------------------------------
# System parameters
# -----------------------------------------------------
g = 9.81
l = 1
M = 10
m = 1
k = 20

# -----------------------------------------------------
# Initial conditions
# -----------------------------------------------------
x0 = 1.6
th0 = np.radians(170)
v0, ome0 = 0, 0

# -----------------------------------------------------
# Method parameters
# -----------------------------------------------------
tmax = 30          # simulated time
dt = 0.005         # RK4 time step
STRIDE = 10        # display 1 every STRIDE steps

# -----------------------------------------------------
# Dynamics: Spring-cart pendulum
# -----------------------------------------------------
def dyn(t, y):
    x, v, th, ome = y
    den = M/m + sin(th)**2
    ax = ((g*cos(th) + l*ome**2)*sin(th) - (k/m)*x) / den
    a_th = -(1/l) * (g*(1 + M/m)*sin(th) + cos(th)*(l*ome**2*sin(th) - (k/m)*x)) / den
    return np.array([v, ax, ome, a_th])

# -----------------------------------------------------
# Fourth-order Runge-Kutta method
# -----------------------------------------------------
def rk4(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + h/2, y + k1/2)
    k3 = h * f(t + h/2, y + k2/2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# -----------------------------------------------------
# Integration using RK4
# -----------------------------------------------------
n = int(tmax / dt)
t = np.linspace(0, n * dt, n + 1)
y = np.empty((n + 1, 4))
y[0] = np.array([x0, v0, th0, ome0])
for i in range(n):
    y[i + 1] = rk4(dyn, t[i], y[i], dt)

# -----------------------------------------------------
# Separate variables after integration
# -----------------------------------------------------
x   = y[:, 0]
v   = y[:, 1]
th  = y[:, 2]
ome = y[:, 3]

# -----------------------------------------------------
# Kinematics
# -----------------------------------------------------
xp = x - l * sin(th)
yp = -l * cos(th)

# -----------------------------------------------------
# Figure
# -----------------------------------------------------
from matplotlib.patches import Rectangle
fig, ax = plt.subplots(figsize=(8, 6))
R = 2.8
ax.set(xlim=(-R, R), ylim=(-l - 0.2,l + 0.2), aspect="equal", title="Spring-cart pendulum (RK4)")
ax.title.set_fontsize(18)
ax.tick_params(axis="both", labelsize=12)
ax.grid(alpha=0.3)

# -----------------------------------------------------
# Wall
# -----------------------------------------------------
wall_x = -2.5
ax.plot([wall_x, wall_x],[-0.15, 0.65], lw=3, color="black")

# -----------------------------------------------------
# Cart dimensions
# -----------------------------------------------------
cart_width = 0.50
cart_height = 0.30

# Cart centered vertically around y = 0
cart = Rectangle((x[0] - cart_width/2, -cart_height/2), cart_width, 
            cart_height, facecolor="lightgray", edgecolor="black", lw=2)
ax.add_patch(cart)

# -----------------------------------------------------
# Pendulum
# -----------------------------------------------------
line, = ax.plot([], [], "o-", lw=1.5, color="blue")

# -----------------------------------------------------
# Trace
# -----------------------------------------------------
trace, = ax.plot([], [], "-", lw=1, alpha=0.6, color="red")

# -----------------------------------------------------
# Spring
# -----------------------------------------------------
spring, = ax.plot([], [], lw=1.5, color="black")


# -----------------------------------------------------
# Clock
# -----------------------------------------------------
clock = ax.text(0.05, 0.93, "", transform=ax.transAxes, fontsize=12)

# -----------------------------------------------------
# Animation function
# -----------------------------------------------------
def animate(i):
    # -------------------------------------------------
    # Move cart
    # -------------------------------------------------
    cart.set_xy((x[i] - cart_width/2, -cart_height/2))

    # -------------------------------------------------
    # Pendulum
    # -------------------------------------------------
    line.set_data([x[i], xp[i]], [0, yp[i]])

    # -------------------------------------------------
    # Trace
    # -------------------------------------------------
    trace.set_data(xp[:i+1], yp[:i+1])

    # -------------------------------------------------
    # Spring
    # -------------------------------------------------
    spring_end = x[i] - cart_width/2
    N = 120
    xs = np.linspace(wall_x, spring_end, N)
    amplitude = 0.08
    turns = 8
    ys = amplitude * np.sin(np.linspace(0, 2*np.pi*turns, N))
    spring.set_data(xs, ys)

    # -------------------------------------------------
    # Clock
    # -------------------------------------------------
    clock.set_text(f"t = {t[i]:.1f} s")
    return line, trace, spring, cart, clock

# -----------------------------------------------------
# Create animation
# -----------------------------------------------------
ani = FuncAnimation(fig, animate, frames=range(0, n + 1, STRIDE), 
                interval=STRIDE * dt * 1000, blit=True)
plt.tight_layout()
plt.show()

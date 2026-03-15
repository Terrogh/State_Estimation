import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# ===== SET THESE PARAMETERS =====
radvec = np.array([0.0, 0.0])        # initial position
angle = np.pi/2
vel = 0.1
velvec = np.array([vel*np.cos(angle), vel*np.sin(angle)])    # velocity (units per frame)
# ================================

fig, ax = plt.subplots()
half_width = 10
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.7)
point, = ax.plot([], [], 'ro', markersize=8)
title = ax.set_title(f'Position: ({radvec[0]:.2f}, {radvec[1]:.2f})\nVelocity: ({velvec[0]:.2f}, {velvec[1]:.2f})')

def init():
    point.set_data([], [])
    return point,

def update(t):
    
    global radvec

    radvec += velvec
        
    point.set_data([radvec[0]], [radvec[1]])
    
    ax.set_xlim(radvec[0] - half_width, radvec[0] + half_width)
    ax.set_ylim(radvec[1] - half_width, radvec[1] + half_width)
    title.set_text(f'Position: ({radvec[0]:.2f}, {radvec[1]:.2f})\nVelocity: ({velvec[0]:.2f}, {velvec[1]:.2f})')
    
    return point, title

ani = FuncAnimation(fig, update, frames=200, init_func=init, blit=False, interval=50)

plt.show()
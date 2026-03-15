import matplotlib.pyplot as plt

# Initial point coordinates
x0, y0 = 0, 0

# Create the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_title(f'Point at ({x0}, {y0})')

plt.legend()
plt.show()
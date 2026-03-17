import matplotlib.pyplot as plt
from   matplotlib.animation import FuncAnimation
import numpy as np

# ================================
dtmillisec = 50                      # milliseconds
dtsec = dtmillisec / 1000.0

# Pos

rad_vec = np.array([0.0, 0.0])  

# Cmd_vel
     
vel_angle = 0     
                   
cmd_vel = 1 * dtsec                       
vel_noise = 0.1 * dtsec

vel = cmd_vel + vel_noise * np.random.randn()
vel_vec = np.array([vel * np.cos(vel_angle), vel * np.sin(vel_angle)])  # k Velocity

acc = 0                                                                 # k Acceleration
# Prediction

pred_acc      = np.array([0.0, 0.0])                # [0] - priori, [1] - posteriori
prev_pred_acc = np.array([0.0, 0.0])

pred_vel      = np.array([cmd_vel, cmd_vel])        # k prediction
prev_pred_vel = np.array([cmd_vel, cmd_vel])        # k-1 prediction

# Measurement

meas_noise = 0.01 * dtsec

# Filter

alpha = 0.001
beta = 0.1

# Estimated

est_point_vec = np.array([0.0, 0.0])

# Arrow

arrow_scale = 20.0
# ================================

fig, ax = plt.subplots()
half_width = 10
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5)

real_point, = ax.plot([], [], 'ro', markersize=8)

est_point,  = ax.plot([], [], 'bs', markersize=6)

arrow = ax.quiver(rad_vec[0], rad_vec[1],
                  vel_vec[0] * arrow_scale, vel_vec[1] * arrow_scale,
                  color='red', angles='xy', scale_units='xy', scale=1,
                  width=0.005)

title = ax.set_title(f'Position: ({rad_vec[0]:.2f}, {est_point_vec[0]:.2f})\nCmd_Vel: {cmd_vel / dtsec:.2f} m/s | Vel: {vel / dtsec:.2f} m/s')
ax.legend(loc='upper right')

def init():
    real_point.set_data([], [])
    est_point.set_data ([], [])
    return real_point, est_point

def update_axis():
    ax.set_xlim(rad_vec[0] - half_width, rad_vec[0] + half_width)
    ax.set_ylim(rad_vec[1] - half_width, rad_vec[1] + half_width)
    title.set_text(f'Position: ({rad_vec[0]:.2f}, {est_point_vec[0]:.2f})\nCmd_Vel: {cmd_vel / dtsec:.2f} m/s | Vel: {vel / dtsec:.2f} m/s')
    
def update_point():
    real_point.set_data ([rad_vec[0]],     [rad_vec[1]])
    est_point.set_data  ([est_point_vec[0]], [est_point_vec[1]])
    arrow.set_offsets   ([[rad_vec[0], rad_vec[1]]])
    arrow.set_UVC       (vel_vec[0] * arrow_scale, vel_vec[1] * arrow_scale)
    
# State    
    
def update_state(vel_noise):
    global rad_vec, vel, vel_vec, acc
    
    prev_vel = vel
    
    vel = cmd_vel + vel_noise * np.random.randn()
    vel_vec = np.array([vel * np.cos(vel_angle), vel * np.sin(vel_angle)])
    
    rad_vec += vel_vec
    
    acc = (vel - prev_vel) / dtsec

# Predictions   
    
def update_pred(alpha, beta):
    global pred_acc, prev_pred_acc, pred_vel, prev_pred_vel
    
    pred_vel[0] = prev_pred_vel[1] + dtsec * prev_pred_acc[1]
    pred_acc[0] = prev_pred_acc[1]
    
    pred_acc[1] = (1 - alpha) * pred_acc[0] + alpha * measurement_acc(meas_noise)
    pred_vel[1] = pred_vel[0] + beta * dtsec * pred_acc[1]
    
# Estimation
def update_estimation():
    global est_point_vec, pred_vel
    
    est_point_vec += np.array([pred_vel[1], 0.0])
    
# Main

cnt_t = 0
error_sum = 0

def update_frame(frame):
    global cnt_t, error_sum
    
    update_state(vel_noise)
    
    update_pred(alpha, beta)
    
    update_estimation()
    
    cnt_t += 1
    error_sum += abs(rad_vec[0] - est_point_vec[0])
    print(f"position error = {error_sum:.4f}, time = {cnt_t}")

    update_point()
    update_axis()
    return real_point, arrow, title

# Measurement

def measurement_acc(meas_noise):
    return acc + meas_noise * np.random.randn()

ani = FuncAnimation(fig, update_frame, frames=1, init_func=init, blit=False, interval=dtmillisec)

plt.show()
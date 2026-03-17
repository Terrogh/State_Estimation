import matplotlib.pyplot as plt
from   matplotlib.animation import FuncAnimation
import numpy as np

#1D. no matrixes no shit. state is pos + velocity + acc

class System():
    def __init__(self, dt, cmd_vel, real_state, sigma_cmd_noise, sigma_meas_noise):
        self.dt = dt
        self.cmd_vel = cmd_vel                      
        self.real_state = real_state.copy()
        self.prev_real_state = real_state.copy()
        self.sigma_cmd_noise = sigma_cmd_noise       
        self.sigma_meas_noise = sigma_meas_noise
        
        # Initials (for reset)
        
        self.initial_state = real_state.copy()    
        
    def reset(self):
        self.real_state = self.initial_state.copy()
        self.prev_real_state = self.initial_state.copy()
        
    def update(self):
        self.prev_real_state = self.real_state.copy()
        
        self.real_state[0] = self.prev_real_state[0] + self.prev_real_state[1] * self.dt                                         # Position
        self.real_state[1] = self.cmd_vel + self.sigma_cmd_noise * np.random.randn()                                             # Commanded Velocity
        self.real_state[2] = (self.real_state[1] - self.prev_real_state[1]) / self.dt                                            # Accurate Acceleration
        
from collections import deque   

class AlphaBetaGammaFilter():
    def __init__(self, dt, abg, system, initial_state, improved = False):
        # Animation 
        self.dt = dt
        
        # Filter params (abg - alpha, beta, gamma)
        self.alpha = abg[0]
        self.beta = abg[1]
        self.gamma = abg[2]
        
        # System
        self.system = system

        # Predictions
        self.priori_state  = initial_state.copy()                                                                                        # k priori state
        self.posteriori_state = initial_state.copy()                                                                                     # k posteriori state
                                    
        self.prev_priori_state  = initial_state.copy()                                                                                   # k-1 priori state
        self.prev_posteriori_state = initial_state.copy()                                                                                # k-1 posteriori state
        
        # Initials (for reset)
        self.initial_state = initial_state.copy()
        
        # + buffer and mean of the buffer
        self.improved = improved
        if self.improved:
            self.data = {
                'measurements'  : deque(maxlen=10),
                'meas_mean'     : initial_state[2],
                'accelerations' : deque(maxlen=10),
                'acc_mean'      : initial_state[2]
            }
            self.update = self._update_imprvd
        else:
            self.update = self._update_std
            
    def reset(self):
        self.priori_state  = self.initial_state.copy()                                                                                        
        self.posteriori_state = self.initial_state.copy()                                                                                     
                                    
        self.prev_priori_state  = self.initial_state.copy()                                                                                   
        self.prev_posteriori_state = self.initial_state.copy()                                                                                
    
    def measurement(self):
        return self.system.real_state[2] + self.system.sigma_meas_noise * np.random.randn()                                         # Measured Velocity
    
    def predict(self):
        self.prev_posteriori_state = self.posteriori_state.copy()
        self.prev_priori_state = self.priori_state.copy()
        
        self.priori_state[2] = self.prev_posteriori_state[2]                                                                        # Predict Acceleration
        self.priori_state[1] = self.prev_posteriori_state[1] + self.dt * self.prev_posteriori_state[2]                              # Predict Velocity
        self.priori_state[0] = self.prev_posteriori_state[0] + self.dt * self.prev_posteriori_state[1] + (self.dt**2 / 2) * self.prev_posteriori_state[2]       # Predict Position
        
    def _update_std(self):
        measured_acc = self.measurement()
        
        self.posteriori_state[2] = self.priori_state[2] * (1 - self.alpha) + self.alpha * measured_acc                              # Update Acceleration
        self.posteriori_state[1] = self.priori_state[1] + self.beta * self.dt * (measured_acc - self.priori_state[2])               # Update Velocity
        self.posteriori_state[0] = self.priori_state[0] + self.gamma * (self.dt**2 / 2) * (measured_acc - self.priori_state[2])     # Update Position
    
    def _update_imprvd(self):
        measured_acc = self.measurement()
        # self.data_upd(measured_acc, self.posteriori_state[2])
        
        self.posteriori_state[2] = self.data['acc_mean'] * (1 - self.alpha) + self.alpha * self.data['meas_mean']                              # Update Acceleration
        self.posteriori_state[1] = self.priori_state[1] + self.beta * self.dt * (self.data['meas_mean']  - self.data['acc_mean'])               # Update Velocity
        self.posteriori_state[0] = self.priori_state[0] + self.gamma * (self.dt**2 / 2) * (self.data['meas_mean']  - self.data['acc_mean'])     # Update Position
        
    def data_upd(self, meas_acc, acc):
        self.data['measurements'].append(meas_acc)
        self.data['meas_mean'] = sum(self.data['measurements']) / len(self.data['measurements'])
        
        self.data['accelerations'].append(acc)
        self.data['acc_mean'] = sum(self.data['accelerations']) / len(self.data['accelerations'])
        
class Experiment:
    def __init__(self, system, filter, record=True):
        self.system = system
        self.filter = filter
        self.record = record

        if self.record:
            self.data = {
                't': [],
                'true_pos': [],
                'true_vel': [],
                'est_pos': [],
                'est_vel': []
            }
            
    def reset(self):
        self.system.reset()
        self.filter.reset()
        
        if self.record:
            self.data = {
                't': [],
                'true_pos': [],
                'true_vel': [],
                'est_pos': [],
                'est_vel': []
            }
    def step(self, frame):
        """Advance one time step and return current states."""
        self.system.update()
        self.filter.predict()
        self.filter.update()

        true_x = self.system.real_state[0]
        true_v = self.system.real_state[1]
        est_x = self.filter.posteriori_state[0]
        est_v = self.filter.posteriori_state[1]
        alpha = self.filter.alpha

        if self.record:
            t = frame * self.system.dt
            self.data['t'].append(t)
            self.data['true_pos'].append(true_x)
            self.data['true_vel'].append(true_v)
            self.data['est_pos'].append(est_x)
            self.data['est_vel'].append(est_v)

        return true_x, true_v, est_x, est_v, alpha
    
    def error_measure(self):
        true_pos = np.array(self.data['true_pos'])
        est_pos = np.array(self.data['est_pos'])
        pos_error = true_pos - est_pos
        return np.sqrt(np.mean(pos_error**2))

    def run(self, frames):
        """Run the experiment for a given number of frames (no animation)."""
        for frame in range(frames):
            self.step(frame)
        return self.data if self.record else None
    
    def multiplerun(self, runs_number, frames_for_run):
        rmse_array = []
        for run in range(runs_number):
            self.reset()
            self.run(frames_for_run)
            rmse = Experiment.error_measure(self)
            rmse_array.append(rmse)
            print(f"Run {run+1}/{runs_number} completed, RMSE = {rmse:.4f}")
        return rmse_array
            


class Animator:
    def __init__(self, experiment, half_width=10, arrow_scale=5.0):
        self.exp = experiment
        self.half_width = half_width
        self.arrow_scale = arrow_scale

        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.set_ylim(-half_width, half_width)

        # Plot elements
        self.true_point, = self.ax.plot([], [], 'ro', markersize=8, label='True')
        self.est_point,  = self.ax.plot([], [], 'bs', markersize=6, label='Estimated')

        # Arrows
        self.true_arrow = self.ax.quiver(
            0, 0, 0, 0, color='red', angles='xy', scale_units='xy', scale=1,
            width=0.005, label='True velocity'
        )
        self.est_arrow = self.ax.quiver(
            0, 0, 0, 0, color='blue', angles='xy', scale_units='xy', scale=1,
            width=0.005, alpha=0.7, label='Estimated velocity'
        )

        self.title = self.ax.set_title('')
        self.ax.legend(loc='upper right')
        self.ani = None

    def init(self):
        self.true_point.set_data([], [])
        self.est_point.set_data([], [])
        return self.true_point, self.est_point

    def update_plot(self, frame):
        # Let the experiment advance and record data
        true_x, true_v, est_x, est_v, alpha = self.exp.step(frame)

        # Update graphical elements
        self.true_point.set_data([true_x], [0])
        self.est_point.set_data([est_x], [0])

        self.true_arrow.set_offsets([[true_x, 0]])
        self.true_arrow.set_UVC(true_v * self.arrow_scale, 0)

        self.est_arrow.set_offsets([[est_x, 0]])
        self.est_arrow.set_UVC(est_v * self.arrow_scale, 0)

        self.ax.set_xlim(true_x - self.half_width, true_x + self.half_width)

        self.title.set_text(
            f'True: x={true_x:.2f} m, v={true_v:.2f} m/s  |  '
            f'Est: x={est_x:.2f} m, v={est_v:.2f} m/s  |  '
            f'Alpha: {alpha:.2f}'
        )

        return self.true_point, self.est_point, self.true_arrow, self.est_arrow, self.title

    def animate(self, frames=500, interval=None):
        if interval is None:
            interval = self.exp.system.dt * 1000
        self.ani = FuncAnimation(
            self.fig, self.update_plot, frames=frames,
            init_func=self.init, blit=False, interval=interval
        )
        plt.show()
        # After closing the window, we can optionally call a plotter
        


import matplotlib.pyplot as plt
import numpy as np

class Plotter:    
    @staticmethod
    def plot_experiment(experiment):
        """Plot the recorded data from an Experiment object."""
        if not experiment.record or len(experiment.data['t']) == 0:
            print("No data recorded.")
            return

        t = np.array(experiment.data['t'])
        true_pos = np.array(experiment.data['true_pos'])
        est_pos = np.array(experiment.data['est_pos'])
        true_vel = np.array(experiment.data['true_vel'])
        est_vel = np.array(experiment.data['est_vel'])

        pos_error = est_pos - true_pos
        rmse = np.sqrt(np.mean(pos_error**2))

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        ax1.plot(t, true_pos, 'r-', label='True')
        ax1.plot(t, est_pos, 'b--', label='Estimated')
        ax1.set_ylabel('Position (m)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(t, true_vel, 'r-', label='True')
        ax2.plot(t, est_vel, 'b--', label='Estimated')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3.plot(t, pos_error, 'g-', label='Error')
        ax3.axhline(0, color='black', linewidth=0.5)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Error (m)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_title(f'Position error (RMSE = {rmse:.4f} m)')

        plt.tight_layout()
        plt.show()
        
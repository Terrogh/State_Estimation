import matplotlib.pyplot as plt
from   matplotlib.animation import FuncAnimation
import numpy as np

#1D. state is pos + velocity + acc

class System():
    def __init__(self, dt, cmd_vel, real_state, sigma_cmd_noise, sigma_meas_noise):
        self.dt = dt
        
        # Command Stuff
        self.cmd_vel_start_val = cmd_vel
        self.cmd_vel = cmd_vel
        self.prev_cmd_vel = cmd_vel
        
        # States              
        self.real_state = (np.matrix(real_state)).T
        self.prev_real_state = self.real_state.copy()
        
        # Noises
        self.sigma_cmd_noise = sigma_cmd_noise       
        self.sigma_meas_noise = sigma_meas_noise
        
        # Initials (for reset)
        self.initial_state = self.real_state.copy()
        self.t = 0
        
    def reset(self):
        self.real_state = self.initial_state.copy()
        self.prev_real_state = self.initial_state.copy()
        
    def update(self):
        self.prev_cmd_vel = self.cmd_vel
        self.t += self.dt
        self.cmd_vel = self.cmd_vel_start_val* np.sin(self.t) + 1
        self.prev_real_state = self.real_state.copy()
        
        self.real_state[0] = self.prev_real_state[0] + self.prev_real_state[1] * self.dt                                           # Accurate Position
        self.real_state[1] = self.cmd_vel + self.sigma_cmd_noise * np.random.randn()                                               # Commanded Velocity
        self.real_state[2] = (self.real_state[1] - self.prev_real_state[1]) / self.dt                                              # Accurate Acceleration  

class AlphaBetaGammaFilter():
    def __init__(self, dt, abg, system, initial_state):
        # Step Time 
        self.dt = dt
        
        # System
        self.system = system
        
        # Alpha Beta Gamma (some kind of)
        self.abg = abg

        # Predictions
        self.priori_state  = (np.matrix(initial_state)).T                                                                           # k priori state
        self.posteriori_state = self.priori_state.copy()                                                                            # k posteriori state
             
        self.prev_priori_state  = self.priori_state.copy()                                                                          # k-1 priori state
        self.prev_posteriori_state = self.priori_state.copy()                                                                       # k-1 posteriori state
        
        # Indicator Matrix
        self.I = np.matrix(np.eye(3))
        
        # Error Matrixes
        self.Q = np.matrix([[dt ** 2,  dt, 2 * dt],
                            [     dt,   1,      2],
                            [ 2 * dt,   2,      4]]) * system.sigma_cmd_noise ** 2
        self.R = system.sigma_meas_noise ** 2
        
        # State Matrix and Input Matrix
        self.F = np.matrix([[1, 0, 0], 
                            [0, 0, 0], 
                            [0, 0, 0]])
        self.F_transposed = (self.F).T
        
        self.G = np.matrix([[0, dt, 0],
                            [0, 1,  0],
                            [0, 0,  1]])
        
        self.U = (np.matrix([0, self.system.cmd_vel, (self.system.cmd_vel - self.system.prev_cmd_vel) / self.dt])).T
        
        # Kalman Gain Matrix
        self.KG = (np.matrix([abg[0] * (self.dt**2 / 2), abg[1] * self.dt, abg[2]])).T
        
        # Measurement Matrix
        self.H = np.matrix([0, 0, 1])
        self.H_transposed = (self.H).T
        
        self.meas = np.dot(self.H, self.system.real_state.copy()) + self.system.sigma_meas_noise * np.random.randn()
        
        # Estimation Error Cov
        self.priori_P = self.I  * system.sigma_meas_noise ** 2
        self.posteriori_P = np.dot((self.I - np.dot(self.KG, self.H)), self.priori_P)
        
        # Initials (for reset)
        self.initial_state = initial_state.copy()
            
    def reset(self):
        self.priori_state  = self.initial_state.copy()                                                                                        
        self.posteriori_state = self.initial_state.copy()                                                                                     
                                    
        self.prev_priori_state  = self.initial_state.copy()                                                                                   
        self.prev_posteriori_state = self.initial_state.copy()                                                                                
    
    def measurement(self):
        self.meas = self.system.real_state[2] + self.system.sigma_meas_noise * np.random.randn()                                                                                              
    
    def U_upd(self):
        self.U[1] =  self.system.cmd_vel
        self.U[2] = (self.system.cmd_vel - self.system.prev_cmd_vel) / self.dt
    
    def predict(self):
        self.prev_posteriori_state = self.posteriori_state.copy()
        self.prev_priori_state = self.priori_state.copy()
        self.U_upd()
        
        self.priori_state = np.dot(self.F, self.prev_posteriori_state) + np.dot(self.G, self.U)
    
    def update(self):
        self.measurement()
        self.data_upd()
        innovation = self.meas - np.dot(self.H, self.priori_state)
        
        self.posteriori_state = self.priori_state + np.dot(self.KG, innovation)
        
        self.abg[0] = self.KG[0,0] / (self.dt**2 / 2)
        self.abg[1] = self.KG[1,0] / (self.dt)
        self.abg[2] = self.KG[2,0]
        
    def data_upd(self):
        # print("priori P:")
        # print(self.priori_P)
        # print("posteriori P:")
        # print(self.posteriori_P)
        # print("KG:")
        # print(self.KG)
        # print("Estimated State:")
        # print(self.posteriori_state)
        self.priori_P = np.dot(self.F, np.dot(self.posteriori_P, self.F_transposed)) + self.Q
        self.posteriori_P = np.dot((self.I - np.dot(self.KG, self.H)), self.priori_P)
        self.KG = np.dot(self.posteriori_P, np.dot(self.H_transposed, self.R))
        
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
                'true_a'  : [],
                'est_pos': [],
                'est_vel': [],
                'est_a'  : []
            }
            
    def reset(self):
        self.system.reset()
        self.filter.reset()
        
        if self.record:
            self.data = {
                't': [],
                'true_pos': [],
                'true_vel': [],
                'true_a'  : [],
                'est_pos': [],
                'est_vel': [],
                'est_a'  : []
            }
    def step(self, frame):
        """Advance one time step and return current states."""
        self.system.update()
        self.filter.predict()
        self.filter.update()

        true_x = self.system.real_state[0, 0]
        true_v = self.system.real_state[1, 0]
        true_a = self.system.real_state[2, 0]
        est_x = self.filter.posteriori_state[0, 0]
        est_v = self.filter.posteriori_state[1, 0]
        est_a = self.filter.posteriori_state[2, 0]
        alpha = self.filter.abg[0]
        beta = self.filter.abg[1]
        gamma = self.filter.abg[2]

        if self.record:
            t = frame * self.system.dt
            self.data['t'].append(t)
            self.data['true_pos'].append(true_x)
            self.data['true_vel'].append(true_v)
            self.data['true_a'].append(true_a)
            self.data['est_pos'].append(est_x)
            self.data['est_vel'].append(est_v)
            self.data['est_a'].append(est_a)

        return true_x, true_v, true_a, est_x, est_v, est_a, alpha, beta, gamma
    
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
        true_x, true_v, true_a, est_x, est_v, est_a, alpha, beta, gamma = self.exp.step(frame)

        # Update graphical elements
        self.true_point.set_data([true_x], [0])
        self.est_point.set_data([est_x], [0])

        self.true_arrow.set_offsets([[true_x, 0]])
        self.true_arrow.set_UVC(true_v * self.arrow_scale, 0)

        self.est_arrow.set_offsets([[est_x, 0]])
        self.est_arrow.set_UVC(est_v * self.arrow_scale, 0)

        self.ax.set_xlim(true_x - self.half_width, true_x + self.half_width)

        self.title.set_text(
            f'True: x={true_x:.2f} m, v={true_v:.2f} m/s, a={true_a:.2f}  |  '
            f'Est: x={est_x:.2f} m, v={est_v:.2f} m/s, a={est_a:.2f}  |  \n'
            f'alpha: {alpha:.2f}'
            f'beta:  {beta:.2f}'
            f'gamma: {gamma:.2f}'
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
        true_a = np.array(experiment.data['true_a'])
        est_a = np.array(experiment.data['est_a'])

        pos_error = est_pos - true_pos
        rmse = np.sqrt(np.mean(pos_error**2))

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

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
        
        ax3.plot(t, true_a, 'r-', label='True')
        ax3.plot(t, est_a, 'b--', label='Estimated')
        ax3.set_ylabel('Acceleration (m/s^2)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4.plot(t, pos_error, 'g-', label='Error')
        ax4.axhline(0, color='black', linewidth=0.5)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Error (m)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_title(f'Position error (RMSE = {rmse:.4f} m)')
        

        plt.tight_layout()
        plt.show()
        
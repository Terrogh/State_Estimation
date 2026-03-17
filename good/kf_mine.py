import matplotlib.pyplot as plt
from   matplotlib.animation import FuncAnimation
import numpy as np

#1D. no matrixes no shit. state is pos + velocity + acc

class System():
    def __init__(self, dt, cmd_vel, real_state, prev_real_state, sigma_cmd_noise, sigma_meas_noise):
        self.dt = dt
        self.cmd_vel = cmd_vel
        self.real_state = real_state
        self.prev_real_state = prev_real_state
        self.sigma_cmd_noise = sigma_cmd_noise
        self.sigma_meas_noise = sigma_meas_noise
        
    def update(self):
        self.prev_real_state = self.real_state.copy()
        
        self.real_state[0] = self.prev_real_state[0] + self.prev_real_state[1] * self.dt                     # Position
        self.real_state[1] = self.cmd_vel + self.sigma_cmd_noise * np.random.randn()                         # Commanded Velocity
        self.real_state[2] = (self.real_state[1] - self.prev_real_state[1]) / self.dt                        # Accurate Acceleration
           

class AlphaBettaFilter():
    def __init__(self, dt, alpha, beta, system, cmd_vel):
        # Animation 
        self.dt = dt
        
        # Filter params
        self.alpha = alpha
        self.beta = beta
        
        # Initial state
        self.system = system
        self.cmd_vel = cmd_vel

        # Predictions
        self.priori_state  = system.real_state.copy()                                                                # k priori state
        self.posteriori_state = system.real_state.copy()                                                             # k posteriori state
                
        self.prev_priori_state  = self.prev_real_state.copy()                                                        # k-1 priori state
        self.prev_posteriori_state = self.prev_real_state.copy()                                                     # k-1 posteriori state
        
    def measurement(self):
        return self.system.real_state[2] + self.system.sigma_meas_noise * np.random.randn()                   # Measured Velocity
    
    def predict(self):
        self.prev_posteriori_state = self.posteriori_state.copy()
        self.prev_priori_state = self.priori_state.copy()
        
        self.priori_state[2] = self.prev_posteriori_state[2]                                                  # Predict Acceleration
        self.priori_state[1] = self.prev_posteriori_state[1] + self.dt * self.prev_posteriori_state[2]        # Predict Velocity
        self.priori_state[0] = self.prev_posteriori_state[0] + self.dt * self.prev_posteriori_state[1]        # Predict Position
        
    def update(self):
        self.posteriori_state[1] = self.priori_state[1] + self.beta * self.dt * self.priori_state[2]          # Update Velocity
        self.posteriori_state[2] = (1 - self.alpha) * self.priori_state[2] + self.alpha * self.measurement()  # Update Acceleration
        self.posteriori_state[0] = self.prev_posteriori_state[0] + self.dt * self.posteriori_state[1]         # Update Position
    
    # #TODO split this function to update and predict.    
    # def prediction_update(self, real_vel, prev_real_vel): 
    #     self.pred_vel[0] = self.prev_pred_vel[1] + self.dtsec * self.prev_pred_acc[1]
    #     self.pred_acc[0] = self.prev_pred_acc[1]

    #     self.pred_acc[1] = (1 - self.alpha) * self.pred_acc[0] + self.alpha * self.measurement(real_vel, prev_real_vel)
    #     self.pred_vel[1] = self.pred_vel[0] + self.beta * self.dtsec * self.pred_acc[1]
    
class Animator:
    def __init__(self, system, filter, half_width=10, arrow_scale=20.0):
        """
        Parameters
        ----------
        system : System
            The true system instance.
        filter : AlphaBettaFilter
            The filter instance.
        half_width : float
            Half the width of the view window (the camera follows the true point).
        arrow_scale : float
            Scaling factor for velocity arrows (makes them visible).
        """
        self.system = system
        self.filter = filter
        self.half_width = half_width
        self.arrow_scale = arrow_scale

        # Set up the figure
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.set_ylim(-half_width, half_width)      # fixed y-range, motion along x

        # True point (red circle)
        self.true_point, = self.ax.plot([], [], 'ro', markersize=8, label='True')

        # Estimated point (blue square)
        self.est_point, = self.ax.plot([], [], 'bs', markersize=6, label='Estimated')

        # True velocity arrow (attached to true point)
        true_vel_x = self.system.real_state[1]
        self.true_arrow = self.ax.quiver(
            self.system.real_state[0], 0,
            true_vel_x * self.arrow_scale, 0,
            color='red', angles='xy', scale_units='xy', scale=1,
            width=0.005, label='True velocity'
        )

        # Estimated velocity arrow (attached to estimated point, dashed)
        est_vel_x = self.filter.posteriori_state[1]
        self.est_arrow = self.ax.quiver(
            self.filter.posteriori_state[0], 0,
            est_vel_x * self.arrow_scale, 0,
            color='blue', angles='xy', scale_units='xy', scale=1,
            width=0.005, linestyle='dashed', alpha=0.7, label='Estimated velocity'
        )

        self.title = self.ax.set_title('')
        self.ax.legend(loc='upper right')

        # Animation object (will be created later)
        self.ani = None

    def init(self):
        """Initialize the plot elements."""
        self.true_point.set_data([], [])
        self.est_point.set_data([], [])
        return self.true_point, self.est_point

    def update_plot(self, frame):
        """Update function called for each animation frame."""
        # Step 1: Advance the true system
        self.system.update()

        # Step 2: Run the filter
        self.filter.predict()
        self.filter.update()

        # Get current values
        true_x = self.system.real_state[0]
        true_v = self.system.real_state[1]
        est_x  = self.filter.posteriori_state[0]
        est_v  = self.filter.posteriori_state[1]

        # Update points (they move along y=0)
        self.true_point.set_data([true_x], [0])
        self.est_point.set_data([est_x], [0])

        # Update arrows
        self.true_arrow.set_offsets([[true_x, 0]])
        self.true_arrow.set_UVC(true_v * self.arrow_scale, 0)

        self.est_arrow.set_offsets([[est_x, 0]])
        self.est_arrow.set_UVC(est_v * self.arrow_scale, 0)

        # Center the camera on the true point
        self.ax.set_xlim(true_x - self.half_width, true_x + self.half_width)

        # Update title
        self.title.set_text(
            f'True: x={true_x:.2f} m, v={true_v:.2f} m/s  |  '
            f'Est: x={est_x:.2f} m, v={est_v:.2f} m/s\n'
            f'True accel: {self.system.real_state[2]:.3f} m/s²'
        )

        return self.true_point, self.est_point, self.true_arrow, self.est_arrow, self.title

    def animate(self, frames=500, interval=None):
        """Start the animation."""
        if interval is None:
            interval = self.system.dt * 1000   # convert dt to milliseconds
        self.ani = FuncAnimation(
            self.fig, self.update_plot, frames=frames,
            init_func=self.init, blit=False, interval=interval
        )
        plt.show()
        return self.ani

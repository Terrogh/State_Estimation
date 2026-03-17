import matplotlib.pyplot as plt
from   matplotlib.animation import FuncAnimation
import numpy as np

#2D cords and velocity no matrixes no shit state is cords + velocity + acc

class AlphaBettaFilter():
    def init(self, dt, alpha, beta, cords, velocity, cmd_vel, meas_acc, sigma_cmd_noise, sigma_meas_noise):
        # Animation 
        self.dt = dt
        # Filter params
        self.alpha = alpha
        self.beta = beta
        # Initial state
        self.cords = cords
        self.velocity = velocity
        self.cmd_vel = cmd_vel
        self.meas_acc = meas_acc
        # Constant? noise
        self.sigma_cmd_noise = sigma_cmd_noise
        self.sigma_meas_noise = sigma_meas_noise
        # Predictions
        self.priori_state  = np.array([cmd_vel, 0.0])                    
        self.posteriori_state = np.array([cmd_vel, 0.0])
        
        self.prev_priori_state  = np.array([cmd_vel, 0.0])                    
        self.prev_posteriori_state = np.array([cmd_vel, 0.0])
        
        #TODO Delete following code
        
        self.pred_acc      = np.array([0.0, 0.0])                # [0] - priori (prediction), [1] - posteriori (update)
        self.prev_pred_acc = np.array([0.0, 0.0])
        
        self.pred_vel      = np.array([cmd_vel, cmd_vel])        # k prediction
        self.prev_pred_vel = np.array([cmd_vel, cmd_vel])        # k-1 prediction
        # # Measurements
        # self.meas_acc
        
    def measurement(self, real_vel, prev_real_vel):
        return (real_vel - prev_real_vel) / self.dt + self.sigma_meas_noise * np.random.randn()
    
    def predict(self):
        self.priori_state[0] = self.prev_posteriori_state[0] + self.dt * self.prev_posteriori_state[1]
        self.priori_state[1] = self.prev_posteriori_state[1]
        
    def update(self, real_vel, prev_real_vel):
        self.posteriori_state[0] = self.priori_state[0] + self.beta * self.dt * self.priori_state[1]
        self.posteriori_state[1] = (1 - self.alpha) * self.priori_state[1] + self.alpha * self.measurement(real_vel, prev_real_vel)
    
    #TODO split this function to update and predict.    
    def prediction_update(self, real_vel, prev_real_vel): 
        self.pred_vel[0] = self.prev_pred_vel[1] + self.dtsec * self.prev_pred_acc[1]
        self.pred_acc[0] = self.prev_pred_acc[1]

        self.pred_acc[1] = (1 - self.alpha) * self.pred_acc[0] + self.alpha * self.measurement(real_vel, prev_real_vel)
        self.pred_vel[1] = self.pred_vel[0] + self.beta * self.dtsec * self.pred_acc[1]

#TODO make class to work with pyplot
class Animation():
    def init(self):
        self
        
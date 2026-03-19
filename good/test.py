import numpy as np

I = np.matrix(np.eye(3))
R = I * 0.2
R_inv = R.I
H = np.matrix([0, 0, 1])
H_transposed = H.T

print(R_inv)
print(H_transposed)
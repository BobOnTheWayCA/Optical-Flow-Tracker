import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

# Load images
image1 = img.imread("car_1.pgm")
image2 = img.imread("car_2.pgm")

# Convert to float32
A = np.array(image1, dtype='float32')
B = np.array(image2, dtype='float32')

# Temporal gradient
diff = B - A

# Spatial gradients
gX = cv2.Sobel(A, cv2.CV_64F, 1, 0)  # X
gY = cv2.Sobel(A, cv2.CV_64F, 0, 1)  # Y

# Flatten
gX = gX.flatten()
gY = gY.flatten()
diff = -diff.flatten()

# Solve linear system
M = np.column_stack((gX, gY))
Q, R = np.linalg.qr(M, mode='reduced')  # QR decomposition
Qb = np.matmul(Q.T, diff)
u_v = np.linalg.solve(R, Qb)

# Extract motion vector
U = u_v[0]  # X
V = u_v[1]  # Y

# Define center
center_x = image1.shape[1] // 2
center_y = image1.shape[0] // 2
X, Y = np.meshgrid([center_x], [center_y])

# Plot with motion vector
plt.imshow(image1, cmap='gray')
plt.quiver(X, Y, U, V, color='r', units = 'dots', angles='xy', scale_units='xy', scale=None)
plt.title("Optical Flow Vector (Center)")
plt.savefig("optical_flow.png")
plt.clf()

# Print results
print("Difference (flattened):\n", diff)
print("M shape:", M.shape)
print("Motion vector U:", U, "V:", V)

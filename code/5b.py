import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

# File paths
image1_path = "car_3.pgm"
image2_path = "car_3_rs_1.png"

# Grayscale and float32
A = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE).astype('float32')
B = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE).astype('float32')

# Temporal gradient
diff = B - A

# Spatial gradients
gX = cv2.Sobel(A, cv2.CV_64F, 1, 0)  # X
gY = cv2.Sobel(A, cv2.CV_64F, 0, 1)  # Y

# Center of image
height, width = A.shape
center_x = width // 2
center_y = height // 2

# Rotation and scale gradients
I_r = np.zeros_like(A, dtype=np.float64)
I_s = np.zeros_like(A, dtype=np.float64)

for y in range(height):
    for x in range(width):
        dx = x - center_x  # X to center distance
        dy = y - center_y # Y to center distance

        # Rotation gradient: -dy * Ix + dx * Iy
        I_r[y, x] = -dy * gX[y, x] + dx * gY[y, x]

        # Scale gradient: (1 / sqrt(dx^2 + dy^2)) * (dx * Ix + dy * Iy)
        norm = np.sqrt(dx**2 + dy**2)
        if norm != 0:
            I_s[y, x] = (1 / norm) * (dx * gX[y, x] + dy * gY[y, x])
        else:
            I_s[y, x] = 0

# Flatten gradients
I_r = I_r.flatten()
I_s = I_s.flatten()
diff = -diff.flatten()

# Solve linear system
M = np.column_stack((I_r, I_s))
Q, R = np.linalg.qr(M, mode='reduced')  # QR decomposition
Qb = np.matmul(Q.T, diff)
u_v = np.linalg.solve(R, Qb)

# Extract vectors
U = u_v[0]  # Rotation X
V = u_v[1]  # Scale Y

# Center point
X, Y = np.meshgrid([center_x], [center_y])

# Combine images
combined_image = cv2.addWeighted(A, 0.5, B, 0.5, 0)

# Plot vectors
plt.imshow(combined_image, cmap='gray')
plt.quiver(X, Y, V, U, color='r', units='dots', angles='xy', scale_units='xy', scale=None)

# Descriptions on image
rotation_description = "Clockwise Rotation" if U > 0 else "Counter-Clockwise Rotation"
scale_description = "Expanding (Zoom Out)" if V < 0 else "Contracting (Zoom In)"

plt.text(
    0.98, 0.98,
    f"{rotation_description}\n{scale_description}",
    color='white', fontsize=10, ha='right', va='top', transform=plt.gca().transAxes,
    bbox=dict(facecolor='black', alpha=1, edgecolor='none')
)

plt.text(
    0.98, 0.02,
    f"U (Rotation): {U:.5e}\nV (Scale): {V:.5e}",
    color='white', fontsize=10, ha='right', va='bottom', transform=plt.gca().transAxes,
    bbox=dict(facecolor='black', alpha=1, edgecolor='none')
)

# Save plot
plt.title("Optical Flow Vector (Rotation and Scale)")
plt.savefig("optical_flow_rs.png")
plt.clf()

# Print results
print("Difference (flattened):\n", diff)
print("M shape:", M.shape)
print("Motion vector U (Rotation):", U, "V (Scale):", V)
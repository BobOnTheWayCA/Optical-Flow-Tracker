import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

# Create folder for resulting images
output_dir = "optical_flow_patches"
os.makedirs(output_dir, exist_ok=True)

# Load
image1 = img.imread("car_1.pgm")
image2 = img.imread("car_2.pgm")

# Convert float32
A = np.array(image1, dtype='float32')
B = np.array(image2, dtype='float32')

# Temporal gradient
diff = (B - A)

# Spatial gradients
gX = cv2.Sobel(A, cv2.CV_64F, 1, 0)  # X
gY = cv2.Sobel(A, cv2.CV_64F, 0, 1)  # Y


# Test different block sizes
# Avaliable block_size for 640*480 image (common factors): 
# (1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 80, 160)
# Note: Best value is 80 or 160 for this case
for block_size in [8, 10, 16, 20, 32, 40, 80, 160]:

    # Divide gradients into patches
    height, width = A.shape
    u_map = np.zeros((height // block_size, width // block_size))  # U
    v_map = np.zeros((height // block_size, width // block_size))  # V

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            # Extract blocks
            patch_gX = gX[i:i + block_size, j:j + block_size].flatten()
            patch_gY = gY[i:i + block_size, j:j + block_size].flatten()
            patch_diff = -diff[i:i + block_size, j:j + block_size].flatten()

            # Construct M for each patch
            M = np.column_stack((patch_gX, patch_gY))
            
            # Linear system
            try:
                Q, R = np.linalg.qr(M, mode='reduced')  # QR decomposition
                Qb = np.matmul(Q.T, patch_diff)
                u_v = np.linalg.solve(R, Qb)
            except np.linalg.LinAlgError:
                u_v = [0, 0]  # Motion to zero if singular matrix

            # Store vectors
            u_map[i // block_size, j // block_size] = u_v[0]  # X
            v_map[i // block_size, j // block_size] = u_v[1]  # Y

    # Visualization
    x_centers = np.arange(block_size // 2, width, block_size)
    y_centers = np.arange(block_size // 2, height, block_size)
    X, Y = np.meshgrid(x_centers, y_centers)

    plt.imshow(image1, cmap='gray')
    plt.quiver(X, Y, u_map, -v_map, color='r', angles='xy', scale_units='xy', scale=None)
    plt.title(f"Optical Flow with Block Size {block_size}")
    output_path = os.path.join(output_dir, f"optical_flow_patches_blocksize_{block_size}.png")
    plt.savefig(output_path)
    plt.clf()

    print("Motion vector field U (per patch):\n", u_map)
    print("Motion vector field V (per patch):\n", v_map)
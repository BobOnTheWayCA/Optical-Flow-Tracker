import cv2
import numpy as np

def compute_rotation_scale_gradients(image):
    height, width = image.shape
    center_x, center_y = width // 2, height // 2

    # Gradient along X and Y
    gX = cv2.Sobel(image, cv2.CV_64F, 1, 0)  # X
    gY = cv2.Sobel(image, cv2.CV_64F, 0, 1)  # Y

    # Rotation and scale gradients
    I_r = np.zeros_like(image, dtype=np.float64)
    I_s = np.zeros_like(image, dtype=np.float64)

    # Calculation
    for y in range(height):
        for x in range(width):
            dx = x - center_x
            dy = y - center_y

            # Rotation gradient: -dy * Ix + dx * Iy
            I_r[y, x] = -dy * gX[y, x] + dx * gY[y, x]

            # Scale gradient: (1 / sqrt(dx^2 + dy^2)) * (dx * Ix + dy * Iy)
            norm = np.sqrt(dx**2 + dy**2)
            if norm != 0:
                I_s[y, x] = (1 / norm) * (dx * gX[y, x] + dy * gY[y, x])
            else:
                I_s[y, x] = 0

    return I_r, I_s

# Load image
image_path = "car_3.pgm"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Compute gradients
I_r, I_s = compute_rotation_scale_gradients(image)

# Display results
print("Rotation Gradient (I_r):")
print(I_r)
print("\nScale Gradient (I_s):")
print(I_s)

# Normalization
I_r_normalized = cv2.normalize(I_r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
I_s_normalized = cv2.normalize(I_s, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Save and display
cv2.imwrite("rotation_gradient.png", I_r_normalized)
cv2.imwrite("scale_gradient.png", I_s_normalized)
cv2.imshow("Rotation Gradient", I_r_normalized)
cv2.imshow("Scale Gradient", I_s_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()

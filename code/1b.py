import cv2
import numpy as np
import matplotlib.image as img

# Load images
# We only need one picture for this task
image1 = img.imread("car_1.pgm")
image2 = img.imread("car_2.pgm")

# Ensure float32
A = np.array(image1, dtype='float32')
B = np.array(image2, dtype='float32')

# Spatial gradients for A
gX = cv2.Sobel(A, cv2.CV_64F, 1, 0)  # X
gY = cv2.Sobel(A, cv2.CV_64F, 0, 1)  # Y

# Normalize gradients
gX_normalized = cv2.normalize(gX, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
gY_normalized = cv2.normalize(gY, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Display
cv2.imshow("Gradient along X-axis (1b)", gX_normalized)
cv2.imshow("Gradient along Y-axis (1b_2)", gY_normalized)

# Save as images
cv2.imwrite("gradient_x_axis.png", gX_normalized)
cv2.imwrite("gradient_y_axis.png", gY_normalized)

# Wait for key press to close
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print gradient
print("Gradient along X-axis (gX):")
print(gX)

print("Gradient along Y-axis (gY):")
print(gY)

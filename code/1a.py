import cv2
import numpy as np
import matplotlib.image as img

# Load two images
image1 = img.imread("car_1.pgm")
image2 = img.imread("car_2.pgm")

# Convert to grayscale
if len(image1.shape) > 2:
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
if len(image2.shape) > 2:
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Ensure float32
A = np.array(image1, dtype='float32')
B = np.array(image2, dtype='float32')

# Temporal difference
diff = B - A

# Threshold temporal difference
threshold_value = 20
_, thresholded_diff = cv2.threshold(np.abs(diff), threshold_value, 255, cv2.THRESH_BINARY)

# Display both result
cv2.imshow("Temporal Difference", diff / np.max(diff))  # Normalize
cv2.imshow("Thresholded Temporal Difference", thresholded_diff)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Save results
cv2.imwrite("temporal_difference.png", diff)
cv2.imwrite("thresholded_temporal_difference.png", thresholded_diff)

# Print matrices
print("Image 1:\n", A)
print("\nImage 2:\n", B)
print("\nTemporal Difference:\n", diff)
print("\nThresholded Temporal Difference:\n", thresholded_diff)

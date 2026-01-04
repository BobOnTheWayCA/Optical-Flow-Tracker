import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create directory
output_dir = "optical_flow_video"
os.makedirs(output_dir, exist_ok=True)
output_filename = os.path.join(output_dir, "optical_flow_single_window.mp4")

# Initialize webcam
cam = cv2.VideoCapture(0)

# Video writer
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cam.get(cv2.CAP_PROP_FPS) if cam.get(cv2.CAP_PROP_FPS) > 0 else 30)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

# Store previous frame
prev_gray = None

while True:
    # Capture
    ret_val, frame = cam.read()
    if not ret_val:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.array(gray, dtype='float32')

    if prev_gray is not None:
        # Temporal gradient
        diff = gray - prev_gray

        # Spatial gradients
        gX = cv2.Sobel(prev_gray, cv2.CV_64F, 1, 0)  # X
        gY = cv2.Sobel(prev_gray, cv2.CV_64F, 0, 1)  # Y

        # Flatten
        gX = gX.flatten()
        gY = gY.flatten()
        diff_flat = -diff.flatten()

        # Solve linear system
        M = np.column_stack((gX, gY))
        try:
            Q, R = np.linalg.qr(M, mode='reduced')  # QR decomposition
            Qb = np.matmul(Q.T, diff_flat)
            u_v = np.linalg.solve(R, Qb)
        except np.linalg.LinAlgError:
            u_v = [0, 0]  # Handle singular matrix

        # Motion vector
        U = u_v[0]  # X
        V = u_v[1]  # Y

        # Center point
        center_x = frame.shape[1] // 2
        center_y = frame.shape[0] // 2
        X, Y = np.meshgrid([center_x], [center_y])

        # Quiver scale factor
        quiver_scale = 1000

        # Draw motion vector
        start_point = (center_x, center_y)
        end_point = (int(center_x + U * quiver_scale), int(center_y + V * quiver_scale))
        cv2.arrowedLine(frame, start_point, end_point, (0, 255, 0), 2, tipLength=0.2)

    # Show with overlay
    cv2.imshow('Single Window Optical Flow', frame)

    # Write frame to output video
    out.write(frame)

    # Update previous frame
    prev_gray = gray

    # Exit condition
    if cv2.waitKey(1) != -1:
        break

cam.release()
out.release()
cv2.destroyAllWindows()
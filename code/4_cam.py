import cv2
import numpy as np
import os

# Output directory
output_dir = "optical_flow_video"
os.makedirs(output_dir, exist_ok=True)

# Parameters
block_size = 30  # Block size for patches
output_fps = 10  # Frames per second for saved video

# Computation function
def compute_optical_flow(prev_frame, curr_frame, block_size):
    height, width = prev_frame.shape
    gX = cv2.Sobel(prev_frame, cv2.CV_64F, 1, 0)  # X
    gY = cv2.Sobel(prev_frame, cv2.CV_64F, 0, 1)  # Y
    diff = curr_frame - prev_frame  # Temporal gradient

    u_map = np.zeros((height // block_size, width // block_size))
    v_map = np.zeros((height // block_size, width // block_size))

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            # Extract block
            patch_gX = gX[i:i + block_size, j:j + block_size].flatten()
            patch_gY = gY[i:i + block_size, j:j + block_size].flatten()
            patch_diff = diff[i:i + block_size, j:j + block_size].flatten()

            # Solve for motion vector
            M = np.column_stack((patch_gX, patch_gY))
            try:
                Q, R = np.linalg.qr(M, mode='reduced')
                Qb = np.matmul(Q.T, patch_diff)
                u_v = np.linalg.solve(R, Qb)
            except np.linalg.LinAlgError:
                u_v = [0, 0]

            u_map[i // block_size, j // block_size] = u_v[0]
            v_map[i // block_size, j // block_size] = u_v[1]

    return u_map, v_map

# Capture video from webcam
cap = cv2.VideoCapture(0)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = os.path.join(output_dir, "optical_flow_patch.mp4")
out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

print("Press any key to stop and save the video.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute
    u_map, v_map = compute_optical_flow(prev_gray, curr_gray, block_size)

    # Draw motion vectors
    x_centers = np.arange(block_size // 2, width, block_size)
    y_centers = np.arange(block_size // 2, height, block_size)
    for y_idx, y in enumerate(y_centers):
        for x_idx, x in enumerate(x_centers):
            u = u_map[y_idx, x_idx]
            v = v_map[y_idx, x_idx]

            # Debugging
            print(f"Vector at ({x_idx}, {y_idx}): U={u}, V={v}")

            # Ensure start_point and end_point are valid
            start_point = (int(round(x)), int(round(y)))
            end_point = (
                min(width - 1, max(0, int(round(x + u)))),
                min(height - 1, max(0, int(round(y + v))))
            )

            cv2.arrowedLine(frame, start_point, end_point, (0, 255, 0), 2, tipLength=0.2)

    # Show preview
    cv2.imshow("Optical Flow Live", frame)

    # Save frame
    out.write(frame)

    # Update previous frame
    prev_gray = curr_gray

    # Exit when key pressed
    if cv2.waitKey(1) != -1:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Optical flow video saved to {output_path}")
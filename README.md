# Optical Flow Tracker

A live OpenCV Optical Flow project visualizes motion by tracking frame-to-frame changes and overlaying flow vectors on images and videos.

<p align="center">
  <img src="optflow.gif" alt="Optical Flow Tracker demo" width="1000">
</p>

---

## Repository Layout



- `code/`
  - `1a.py` — temporal gradient + thresholding
  - `1b.py` — spatial gradients (x/y)
  - `1c.py` — solve single-window flow + quiver overlay
  - `2.py` — live single-window optical flow (camera)
  - `3.py` — optical flow on patches (tiles)
  - `4_cam.py` — live video capture + optical flow overlay recorder
  - `4_readvideo.py` — run optical flow on a recorded video file
  - `5a.py` — rotation/scale gradients
  - `5b.py` — solve rotation/scale motion on a video (single window)
- `good/` — videos that perform well
- `bad/` — videos that perform poorly
- `optical_flow_patches/ & output_images/` — generated plots, overlays, threshold images, etc.

---


### 1. Single Window Optical Flow

#### 1a) Temporal Image Gradient
- **Use:** `code/1a.py`

**What it does:**
- Reads two consecutive frames (grayscale)
- Computes frame difference
- Thresholds the temporal difference to suppress noise

**Expected output:**
- A thresholded temporal-difference image

---

#### 1b) Spatial Image Gradient
- **Use:** `code/1b.py`

**What it does:**
- Computes horizontal and vertical spatial gradients using forward differences
- Avoids `numpy.gradient`

**Expected output:**
- `Ix` and `Iy` visualization images

---

#### 1c) Solve for Motion Vector (u,v) + Plot
- **Use:** `code/1c.py`

**What it does:**
- Solves for a single flow vector on the whole window
- Plots the motion vector (quiver) at the image center

**Expected output:**
- An overlay/quiver image

---

### 2. Single Window Optical Flow Live
- **Use:** `code/2.py`

**What it does:**
- Captures frames from your camera
- Computes single-window optical flow per frame
- Overlays the vector on the live video

**Expected output:**
- A recorded demo video

---

### 3. Optical Flow on Patches
- **Use:** `code/3.py`

**What it does:**
- Splits gradients into block tiles
- Solves a flow vector per patch
- Plots a grid of quiver vectors (one per tile)

**Expected output:**
- Patch-based quiver overlay images
- Multiple results for different `block_size`

---

### 4. Optical Flow Video

- **Live camera capture + record:**
  - **Use:** `code/4_cam.py`
- **Process a recorded video file:**
  - **Use:** `code/4_readvideo.py`

**Expected output:**
- Output video file with Optical flow field overlaid

---

### 5. Rotation and Scale

#### 5a) Rotation and Scale Gradients
- **Use:** `code/5a.py`

**Expected output:**
- Rotation-gradient visualization
- Scale-gradient visualization

#### 5b) Solve Rotation/Scale Motion (Single Window)
- **Use:** `code/5b.py`

**Expected output:**
- An overlay showing rotation + zoom behavior (single window)

---

## How to Run

### Requirements
- Python 3.9+ recommended
- Packages:
  - `opencv-python`
  - `numpy`
  - `matplotlib`

Install:
- `pip install opencv-python numpy matplotlib`

### Run a script
From the repository root:
- `python code/1a.py`
- `python code/1b.py`
- `python code/1c.py`
- `python code/2.py`
- `python code/3.py`
- `python code/4_cam.py`
- `python code/4_readvideo.py`
- `python code/5a.py`
- `python code/5b.py`

> Most scripts assume hard-coded input paths (like `media/frames/...` or `media/input.mp4`).
> If your filenames differ, change the config variables at the top of the script.

---



## License and Credits

- CMPUT 428/615 Computer Vision Lab 1.1 (2025)
- University of Alberta, Department of Computing Science  
- Student work by **Shijie (Bob) Bu**


<div align="right">

<img src="UofAlbertalogo.svg" alt="University of Alberta Logo" width="330px" style="vertical-align: middle;">
</p>
<p style="margin: 0; font-size: 14px; font-weight: bold;">
Department of Computing Science
</p>
<p style="margin: 0; font-size: 14px; font-weight: bold;">
November 2022, Edmonton, AB, Canada
</p>


</div>


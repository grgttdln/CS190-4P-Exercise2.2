# Image Stitching Panorama Application

**CS190-4P — Exercise 2.2**

This repository contains a Python application that performs **image stitching** to produce a single panoramic image from two input images. Image stitching is widely used in computer vision, photo processing, mapping, and robotics.

---

## 📌 Overview

The script automatically detects keypoints, matches similar features between two overlapping images, computes a homography transformation, and merges them into one seamless panorama.

This project demonstrates:

- Feature detection and description (e.g., SIFT/ORB)
- Feature matching
- Homography estimation
- Image warping
- Panorama blending

---

## 📁 Repository Contents

```
📦 CS190-4P-Exercise2.2
 ┣ 📂 images/
 ┃ ┣ 📂 set1/   # Image pair 1
 ┃ ┣ 📂 set2/   # Image pair 2
 ┃ ┣ 📂 set3/   # Image pair 3
 ┃ ┗ 📂 set4/   # Image pair 4
 ┣ 📜 main.py   # Main stitching script
 ┗ 📜 README.md # You are here!
```

You may use any of the provided folders for testing.

---

## 🧰 Dependencies

Make sure you have the following installed:

```bash
pip install opencv-python numpy
```

Optional enhancements:

```bash
pip install matplotlib
```

---

## 🧠 How It Works (Simplified)

1. **Keypoint Detection**
   Finds distinct points in both images.

2. **Descriptor Extraction**
   Computes feature vectors describing each keypoint.

3. **Feature Matching**
   Matches keypoints between both images.

4. **Homography Estimation**
   Uses RANSAC to compute transformation.

5. **Warp & Blend**
   Warps one image and blends them into a panorama.

---

## ✨ Example Output (Conceptual)

Original Images:

```
[ image A ]  [ image B ]
```

Panorama:

```
[        Stitched Panorama        ]
```

---

## ⚠️ Notes & Tips

- Overlap: Images must share visual overlap for matching.
- Lighting: Similar exposure improves stitch quality.
- Rotation: Try to keep camera rotation minimal.

---

## 🧑‍💻 Author

_This project was developed for CS190-4P class exercises._

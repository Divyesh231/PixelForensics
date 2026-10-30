# PixelForensics

A digital image-forensics prototype built with Python, OpenCV, and Streamlit. It visualizes several handcrafted signals that may help a person inspect an image; it does not determine authenticity or prove tampering.

PixelForensics runs three exploratory checks on a single image without requiring an original reference: JPEG recompression difference (ELA), copy-move-like feature matching, and local high-frequency residual variation. Results are heuristics and should be interpreted alongside other evidence.

---

## Visual Output & Demo

Example app runs on sample images from the included MICC-F220 collection. These screenshots illustrate the interface and outputs; they are not an accuracy benchmark.

### Clean reference example — `CRW_4853_scale.jpg`
The app shows a 34% heuristic risk score and does not find strong indicators in this run. The score is not a probability.
![PixelForensics analysis of CRW_4853_scale.jpg](assets/Screenshot%202026-10-30%20120142.png)

### Copy-move example — `CRW_4853tamp37.jpg`
The app reports copy-move-like matches and a 71% heuristic risk score in this run.
![PixelForensics analysis of CRW_4853tamp37.jpg](assets/Screenshot%202026-10-30%20120204.png)

### Copy-move example — `DSC_0812tamp27.jpg`
The app reports copy-move-like matches and a 63% heuristic risk score in this run.
![PixelForensics analysis of DSC_0812tamp27.jpg](assets/Screenshot%202026-10-30%20120221.png)

---

## Detection Pipeline

```text
Input Image (JPEG/PNG)
       |
       +---> Error Level Analysis (ELA)          --> JPEG quantization delta map
       +---> Copy-Move Detection (SIFT + DBSCAN) --> Vector trajectories & cloned hulls
       +---> Local Residual Variation            --> Median-filter residual variance map
       |
       +---> Heuristic Evidence Fusion            --> JET heatmap + heuristic risk score
```

---

## Core Detection Engines

### 1. JPEG Recompression Difference (ELA)
- **Method**: The image is re-saved in memory at a selected JPEG quality and subtracted from the input. The app displays an amplified difference view and heatmap.
- **Interpretation**: Local differences can have many causes. This implementation does not identify compression generations or establish that a bright region was spliced.
- **Output**: Amplified difference view and false-color heatmap.

### 2. Copy-Move Duplication Detection
- **Method**: Extracts SIFT keypoints (or ORB where SIFT is unavailable), filters self-matches by a ratio threshold and minimum distance, then clusters displacement vectors with DBSCAN.
- **Interpretation**: Clusters are candidate copy-move-like matches; repetitive textures can produce false matches, and the implementation does not verify a geometric transform.
- **Output**: Match trajectories, convex hull contours, and a candidate-region mask.

### 3. Local Residual Variation
- **Method**: A 3x3 median filter produces a residual `|Original - MedianFiltered|`. The code measures local residual standard deviation in sliding windows (default 16x16) and maps deviation from the image-wide median.
- **Interpretation**: This is a simple local residual-variation heuristic. It does not estimate or match a camera-specific PRNU fingerprint; edges, texture, resizing, and compression can also affect it.
- **Output**: Local residual-variation heatmap.

---

## Included Example Dataset (MICC-F220)

The repository comes pre-loaded with a copy of the **MICC-F220** dataset (University of Florence) under `examples/MICC-F220/`:

- Files named `tamp*.jpg` are tampered with copy-move operations (e.g. `DSCN45tamp1.jpg`, `CRW_4853tamp1.jpg`).
- Files named `_scale.jpg` are the original untouched photos (e.g. `DSCN45_scale.jpg`, `CRW_4853_scale.jpg`).
- `groundtruthDB_220.txt` lists the ground truth labels (1 = tampered, 0 = authentic).

The included dataset is useful for manual demonstrations. The repository does not currently include a reproducible benchmark runner or a measured accuracy evaluation, so no accuracy or confidence claims are made here.

## Score interpretation

The dashboard's 0–100 value is a **heuristic risk score**, computed from fixed weighted component scores. It is not calibrated against a validation set, is not the probability that an image was manipulated, and must not be used as forensic proof. “No strong indicators found” means only that these checks did not produce a high score.

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/your-username/pixel-forensics.git
cd pixel-forensics
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. You can select pre-loaded samples from the sidebar or drag and drop any image from `examples/MICC-F220/` under "Upload your own".

### 3. Run Unit Tests

```bash
pytest tests/
```

Expected output:
```text
tests/test_copy_move.py ..    [ 28%]
tests/test_ela.py ...         [ 71%]
tests/test_noise.py ..        [100%]
7 passed
```

---

## Project Structure

```text
pixel-forensics/
├── app.py                  Main Streamlit dashboard and UI
├── requirements.txt        Minimum dependency versions
├── src/
│   ├── ela.py              Error Level Analysis engine
│   ├── copy_move.py        SIFT + DBSCAN copy-move detector
│   ├── noise_analysis.py   Local residual-variation analysis
│   └── visualization.py    Thermal overlays, mask blending, and verdict logic
├── tests/
│   ├── test_ela.py         Unit tests for ELA
│   ├── test_copy_move.py   Unit tests for copy-move detection
│   └── test_noise.py       Unit tests for noise variance estimation
├── examples/
│   ├── MICC-F220/          Included benchmark dataset (220 images + ground truth)
│   ├── sample_clean.jpg    Clean baseline image
│   ├── sample_cloned.jpg   Copy-move test image
│   └── sample_spliced.jpg  Spliced test image
└── docs/
    └── methodology.md      Detailed algorithm descriptions
```

---

## Limitations

- **Social Media Compression**: Platforms like WhatsApp or Instagram compress images heavily (Q < 65), which smooths out noise residuals and dampens ELA differential spikes.
- **Lossless formats**: ELA recompresses every input as JPEG, including PNG and BMP. Its differences on those files do not reveal a prior JPEG history and should not be interpreted as evidence of editing.
- **Image size**: Uploaded images above 12 megapixels are rejected to limit memory and processing costs.
- **Heuristic signals**: ELA and local residual variation are affected by image content and processing. They are not standalone proof of manipulation.
- **Repetitive Textures**: Highly periodic patterns (brick walls, tile floors, fences) can trigger false feature matches. Increasing the minimum spatial distance slider (`min_dist >= 50px`) may reduce some of these matches but does not eliminate false positives.

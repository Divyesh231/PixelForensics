# PixelForensics

A digital image tampering analysis tool built with Python, OpenCV, and Streamlit.

PixelForensics uses passive (blind) image forensics to detect copy-move cloning, image splicing, and localized edits from a single photograph without needing the original unedited reference image.

---

## Visual Output & Demo

Analysis running on the MICC-F220 benchmark dataset:

### 1. Cloned Object Detection — Antique Firearms (`DSC_0812tamp133.jpg`)
![Firearms Tampering Detection](assets/Screenshot%202026-10-27%20165420.png)

### 2. Duplicated Shelf Detection — Bookstore Aisle (`CRW_4853tamp27.jpg`)
![Bookstore Tampering Detection](assets/Screenshot%202026-10-27%20165840.png)

---

## Detection Pipeline

```text
Input Image (JPEG/PNG)
       |
       +---> Error Level Analysis (ELA)          --> JPEG quantization delta map
       +---> Copy-Move Detection (SIFT + DBSCAN) --> Vector trajectories & cloned hulls
       +---> Sensor Noise Inconsistency          --> High-frequency residual variance map
       |
       +---> Multi-Modal Evidence Fusion         --> JET thermal overlay + confidence score
```

---

## Core Detection Engines

### 1. Error Level Analysis (ELA)
- **Principle**: Lossy JPEG compression introduces quantization error across 8x8 DCT blocks. Repeated saves reduce the error delta over time.
- **Method**: The image is re-saved in memory at JPEG quality Q=90 and subtracted from the input. Spliced foreign content belongs to a different compression generation and shines brighter under differential amplification.
- **Output**: Amplified difference view and a JET false-color thermal overlay.

### 2. Copy-Move Duplication Detection
- **Principle**: Duplicated visual patches within the same image share identical local gradient features.
- **Method**: Extracts SIFT keypoints and matches them against the image itself using k-NN with Lowe's ratio test (tau = 0.72). Spatial distance filtering removes adjacent texture matches. Surviving displacement vectors (dx, dy) are clustered using DBSCAN to group points that share consistent spatial translations.
- **Output**: Trajectory lines linking cloned regions, convex hull contours, and a binary detection mask.

### 3. Local Sensor Noise Inconsistency
- **Principle**: Camera sensors imprint uniform high-frequency noise (PRNU and photon shot noise) across an authentic exposure. Spliced elements or generative fills exhibit mismatched local noise statistics.
- **Method**: A 3x3 median filter suppresses noise while keeping structural edges intact. The absolute residual `|Original - Denoised|` is analyzed across sliding windows (default 16x16) using fast box filters to measure local variance deviation against the image-wide median noise.
- **Output**: High-divergence noise deviation heatmap.

---

## Included Benchmark Dataset (MICC-F220)

The repository comes pre-loaded with a copy of the **MICC-F220** dataset (University of Florence) under `examples/MICC-F220/`:

- Files named `tamp*.jpg` are tampered with copy-move operations (e.g. `DSCN45tamp1.jpg`, `CRW_4853tamp1.jpg`).
- Files named `_scale.jpg` are the original untouched photos (e.g. `DSCN45_scale.jpg`, `CRW_4853_scale.jpg`).
- `groundtruthDB_220.txt` lists the ground truth labels (1 = tampered, 0 = authentic).

### Benchmark Results on Sample Pairs

| Test Image | Type | SIFT Matches | Confirmed Clusters | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| `DSCN45tamp1.jpg` | Tampered | 134 pairs | 1 cluster | TAMPERED (100% confidence) |
| `DSCN45_scale.jpg` | Authentic Baseline | 0 pairs | 0 clusters | LIKELY AUTHENTIC |
| `CRW_4853tamp1.jpg` | Tampered | 67 pairs | 2 clusters | TAMPERED (100% confidence) |
| `CRW_4853_scale.jpg`| Authentic Baseline | 6 pairs | 0 clusters | LIKELY AUTHENTIC |
| `DSC_0812tamp1.jpg` | Tampered | 68 pairs | 1 cluster | TAMPERED (100% confidence) |
| `DSC_0812_scale.jpg`| Authentic Baseline | 0 pairs | 0 clusters | LIKELY AUTHENTIC |

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
├── requirements.txt        Pinned dependencies
├── src/
│   ├── ela.py              Error Level Analysis engine
│   ├── copy_move.py        SIFT + DBSCAN copy-move detector
│   ├── noise_analysis.py   Sensor noise variance analysis
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
- **Lossless Formats**: ELA requires lossy JPEG compression history. Native PNG or BMP files that have never been saved as JPEG produce uniform zero error.
- **Repetitive Textures**: Highly periodic patterns (brick walls, tile floors, fences) can trigger false feature matches. Increasing the minimum spatial distance slider (`min_dist >= 50px`) filters these out.

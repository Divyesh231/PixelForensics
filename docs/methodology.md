# PixelForensics — Mathematical & Algorithmic Methodology

## 1. Introduction

Digital image tampering typically falls into three primary categories:
1. **Copy-Move (Cloning):** Copying a patch from an image and pasting it into another location within the same image (e.g. duplicating crowds or concealing incriminating evidence).
2. **Splicing (Compositing):** Extracting an object from a separate image and inserting it into the target frame.
3. **Inpainting / Generative Fill:** Using generative AI or texture synthesis algorithms to erase or alter objects.

PixelForensics applies a multi-modal heuristic analysis combining spatial-frequency transformations, feature geometry, and statistical noise residuals to detect and localize these manipulations without relying on fragile black-box neural networks.

---

## 2. Forensic Engines

### 2.1 Error Level Analysis (ELA)
When a raw image is saved as a JPEG, the image undergoes Discrete Cosine Transform (DCT) block quantization:

$$F_{Q}(u, v) = \text{round}\left(\frac{\text{DCT}(f(x, y))}{Q(u, v)}\right)$$

Every subsequent save operation degrades the high-frequency coefficients further until an equilibrium is reached. If an external element is spliced into a JPEG image, the spliced region is at its **first generation of quantization**, while the background is at its **second (or subsequent) generation**.

PixelForensics recompresses the image at quality $Q_{\text{test}} = 90$ and calculates the absolute residual:

$$\Delta(x, y) = |I_{\text{original}}(x, y) - I_{\text{recompressed}}(x, y)|$$

Regions that diverge significantly in error level relative to the global canvas mean indicate inconsistent compression histories.

### 2.2 Copy-Move Duplication Detection (Amerini et al.)
Natural scenes contain high feature diversity. When an object is duplicated:
1. **Feature Extraction:** Scale-Invariant Feature Transform (SIFT) or ORB descriptors are extracted:
   $$D = \{d_1, d_2, \dots, d_N\}, \quad d_i \in \mathbb{R}^{128}$$
2. **Self-Matching:** Descriptors are cross-matched against themselves using $k$-nearest neighbors ($k=2$).
3. **Lowe's Ratio Test:** Matches are retained only if:
   $$\frac{\|d_i - d_{\text{match1}}\|}{\|d_i - d_{\text{match2}}\|} < \tau_{\text{ratio}} \quad (\tau \approx 0.72)$$
4. **Spatial Displacement Filtering:** Pairs closer than $D_{\text{min}}$ pixels are discarded to eliminate dense repetitive textures (e.g. brick walls, foliage).
5. **DBSCAN Vector Clustering:** Authentic copy-move manipulations share a consistent spatial translation vector:
   $$\vec{v} = (x_2 - x_1, y_2 - y_1)$$
   PixelForensics clusters displacement vectors using Density-Based Spatial Clustering of Applications with Noise (DBSCAN). Random false matches have chaotic translation angles, while cloned objects form high-density clusters in displacement space.

### 2.3 Sensor Noise Inconsistency (Lyu et al.)
Every imaging sensor exhibits Photo-Response Non-Uniformity (PRNU) and photon shot noise. The noise profile across a pristine photograph follows a uniform Gaussian distribution:

$$I(x, y) = S(x, y) + \eta(x, y), \quad \eta \sim \mathcal{N}(0, \sigma^2)$$

When foreign content is spliced in:
1. High-frequency noise is isolated via median denoising:
   $$R(x, y) = |I(x, y) - \text{MedianFilter}_{3\times3}(I(x, y))|$$
2. Local noise variance is computed across a sliding window $W$ of size $k \times k$:
   $$\text{Var}(R) = \frac{1}{|W|}\sum_{(u, v) \in W} R(u, v)^2 - \left(\frac{1}{|W|}\sum_{(u, v) \in W} R(u, v)\right)^2$$
3. Anomalous local deviations from the global median noise level indicate spliced elements or synthetic smoothing.

---

## 3. Multi-Modal Evidence Fusion

Rather than presenting disjointed outputs, PixelForensics calculates an integrated risk score:

$$S_{\text{composite}} = w_{\text{cm}} \cdot S_{\text{cm}} + w_{\text{ela}} \cdot S_{\text{ela}} + w_{\text{noise}} \cdot S_{\text{noise}}$$

Where $w_{\text{cm}} = 0.50$, $w_{\text{ela}} = 0.30$, and $w_{\text{noise}} = 0.20$. Structural feature duplication carries the highest weight because keypoint clustering produces the lowest false-discovery rate in uncompressed domains.

---

## 4. Scientific Limitations (Honest Disclosure)

1. **Re-compression Attenuation:** If an entire tampered image is re-saved at low JPEG quality (e.g., WhatsApp or social media compression $Q < 60$), ELA differentials are significantly smoothed out.
2. **Periodic Textures:** Strongly repetitive textures (e.g., tiled floors, skyscrapers, chain-link fences) can produce dense feature matches that require strict DBSCAN epsilon tuning.
3. **Lossless Images:** ELA is ineffective on native PNGs or BMPs that have never been subject to lossy DCT quantization.

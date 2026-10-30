# PixelForensics: implementation and methodology

## Purpose and scope

PixelForensics is an educational image-forensics prototype. It computes three heuristic indicators and displays them as visual aids: JPEG recompression difference, copy-move-like feature matches, and local high-frequency residual variation. The indicators are not calibrated, and the tool cannot establish that an image is authentic or manipulated.

## 1. JPEG recompression difference (ELA)

The input is converted to RGB, encoded in memory as JPEG at a user-selected quality, decoded, then subtracted from the input:

\[D(x,y,c)=|I(x,y,c)-J_Q(x,y,c)|\]

The UI shows the amplified difference and a heatmap. The scalar ELA indicator is clipped from the spread between the 95th and 50th percentiles of the per-pixel mean RGB difference:

\[s_{ELA}=\operatorname{clip}((P_{95}(d)-P_{50}(d))/18,0,1)\]

This implementation does not identify or segment a pasted region, infer compression generations, or validate a threshold against a dataset. Image content and encoding choices affect the result. For a PNG input, the image is still re-encoded as JPEG for this comparison.

## 2. Copy-move-like feature matching

The implementation converts the RGB image to grayscale and uses SIFT when available, with ORB as a fallback. A brute-force matcher searches each descriptor against the same image. Candidate matches pass a ratio threshold and a minimum keypoint distance. Their displacement vectors are clustered with DBSCAN; clusters are drawn as lines/hulls and expanded into a visualization mask.

The component score is `min(confirmed cluster matches / 25, 1)`. This is a count-based heuristic, not a confidence estimate. There is no reciprocal-match requirement, transform estimation, or dataset-calibrated false-positive threshold. Repeated textures and feature-poor images can produce missed or spurious matches.

## 3. Local residual-variation map

The grayscale image is median-filtered with a 3x3 kernel. The absolute residual is computed as `R = |I - median(I)|`. A box filter estimates local residual variance in the selected window size. The map shows absolute deviation of the local residual standard deviation from its global median, scaled by the 98th percentile.

The scalar indicator is `min(12 * fraction_of_map_pixels_above_180, 1)`. This is a local texture/residual statistic. It does **not** estimate or correlate Photo-Response Non-Uniformity (PRNU), and it is sensitive to edges, texture, compression, resizing, and denoising.

## 4. Fused map and risk score

The fused visualization combines per-pixel maps after normalizing them to their own ranges:

\[M=0.50M_{copy}+0.30M_{ELA}+0.20M_{residual}\]

The dashboard's scalar heuristic score combines component scores with the same weights:

\[R=0.50s_{copy}+0.30s_{ELA}+0.20s_{residual}\]

The UI displays `100 * R` as an integer truncated toward zero. A copy-move-like cluster with at least ten confirmed matches overrides the scalar thresholds; otherwise the app uses 65 and 40 as high-risk and review thresholds. Map normalization and scalar component calculations are different operations; the heatmap is a qualitative visualization and does not encode the scalar score pixel-for-pixel. The percentage is **not a probability** because the weights and component scales have not been calibrated against labeled validation data.

## Limitations and evaluation status

- Whole-image recompression can obscure or create differences; ELA is not a stand-alone splicing detector.
- Copy-move matches can fail on smooth regions and false-match on repetitive patterns.
- Residual variation is not sensor fingerprint analysis and is not specific to tampering.
- Social-media recompression, resizing, denoising, and image content can change all three signals.
- The repository includes MICC-F220 examples but no reproducible benchmark runner or aggregate evaluation metrics. The examples are demonstrations, not measured accuracy evidence.

For research use, evaluate on held-out labeled images, report precision/recall/F1 and false-positive examples, record all settings and dependency versions, and calibrate any probability claims separately.

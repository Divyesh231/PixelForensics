# app.py - PixelForensics main dashboard
# Run with: streamlit run app.py

from io import BytesIO
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import streamlit as st

from src.ela import compute_ela, make_ela_heatmap
from src.copy_move import detect_copy_move
from src.noise_analysis import compute_noise_map
from src.visualization import (
    make_thermal_overlay,
    make_standalone_heatmap,
    draw_mask_overlay,
    fuse_evidence,
    get_verdict,
    image_to_bytes,
)


# -- page setup ----------------------------------------------------------------

st.set_page_config(page_title="PixelForensics", layout="wide")

st.markdown("""
<style>
.verdict-box {
    border-radius: 8px;
    padding: 14px 20px;
    color: white;
    font-weight: bold;
    font-size: 1.3rem;
    text-align: center;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)


# -- generate sample images if they dont exist ---------------------------------

def setup_samples():
    """Create some basic test images so the app works out of the box."""
    samples_dir = Path("./examples")
    samples_dir.mkdir(exist_ok=True)

    clean_path = samples_dir / "sample_clean.jpg"
    cloned_path = samples_dir / "sample_cloned.jpg"
    spliced_path = samples_dir / "sample_spliced.jpg"

    if not clean_path.exists():
        # make a simple test image with some shapes
        img = np.zeros((450, 650, 3), dtype=np.uint8)
        for y in range(450):
            img[y, :] = [int(120 + 70*(y/450)), int(140 + 50*(y/450)), 180]
        cv2.circle(img, (180, 200), 55, (220, 110, 40), -1)
        cv2.rectangle(img, (350, 180), (450, 340), (40, 160, 90), -1)
        # add some realistic camera noise
        noise = np.random.normal(0, 4, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(clean_path, quality=95)

    if not cloned_path.exists():
        # copy-paste a region to simulate copy-move forgery
        arr = np.array(Image.open(clean_path).convert("RGB"))
        patch = arr[145:255, 125:235].copy()
        arr[145:255, 480:590] = patch  # paste it somewhere else
        Image.fromarray(arr).save(cloned_path, quality=92)

    if not spliced_path.exists():
        # paste a completely foreign element (different noise characteristics)
        arr = np.array(Image.open(clean_path).convert("RGB"))
        foreign = np.zeros((100, 100, 3), dtype=np.uint8)
        foreign[:] = (220, 20, 60)
        cv2.putText(foreign, "FAKE", (15, 55),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        # add heavier noise to make it detectable
        fn = np.random.normal(0, 15, foreign.shape).astype(np.int16)
        foreign = np.clip(foreign.astype(np.int16) + fn, 0, 255).astype(np.uint8)
        arr[60:160, 280:380] = foreign
        Image.fromarray(arr).save(spliced_path, quality=88)

    return {
        "Upload your own": None,
        "Sample: Cloned Object": str(cloned_path),
        "Sample: Spliced Element": str(spliced_path),
        "Sample: Clean Photo": str(clean_path),
    }


# -- sidebar -------------------------------------------------------------------

with st.sidebar:
    st.title("PixelForensics")
    st.caption("Image Tampering Analysis")
    st.markdown("---")

    samples = setup_samples()
    choice = st.selectbox("Test image", list(samples.keys()))

    uploaded = None
    if choice == "Upload your own":
        uploaded = st.file_uploader("Choose file",
                                    type=["jpg", "jpeg", "png", "webp", "bmp"])

    st.markdown("---")
    st.subheader("Settings")

    with st.expander("ELA"):
        ela_q = st.slider("JPEG quality", 70, 98, 90)
        ela_amp = st.slider("Amplification", 5.0, 35.0, 15.0, 1.0)

    with st.expander("Copy-Move"):
        cm_algo = st.selectbox("Detector", ["auto", "sift", "orb"])
        cm_dist = st.slider("Min distance (px)", 15.0, 100.0, 35.0, 5.0)
        cm_ratio = st.slider("Ratio threshold", 0.50, 0.85, 0.72, 0.02)

    with st.expander("Noise Analysis"):
        noise_win = st.slider("Window size", 8, 32, 16, 4)

    st.markdown("---")
    st.caption("All processing happens locally on your machine.")


# -- load the image ------------------------------------------------------------

def load_image():
    if uploaded is not None:
        raw = uploaded.read()
        pil = Image.open(BytesIO(raw)).convert("RGB")
        return np.array(pil, dtype=np.uint8), uploaded.name
    elif choice != "Upload your own" and samples[choice]:
        pil = Image.open(samples[choice]).convert("RGB")
        return np.array(pil, dtype=np.uint8), Path(samples[choice]).name
    return None, None


img, fname = load_image()

if img is None:
    st.header("PixelForensics")
    st.write("Upload an image or pick a sample from the sidebar to start analysis.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("ELA")
        st.write("Finds regions with mismatched JPEG compression history.")
    with c2:
        st.subheader("Copy-Move")
        st.write("Detects duplicated patches via keypoint matching.")
    with c3:
        st.subheader("Noise")
        st.write("Spots foreign elements through noise variance differences.")
    st.stop()


# -- run all engines -----------------------------------------------------------

with st.spinner(f"Analyzing {fname}..."):
    # ELA
    ela_amp_img, ela_raw, ela_score = compute_ela(img, quality=ela_q, scale=ela_amp)
    ela_heatmap = make_ela_heatmap(ela_raw)
    ela_overlay = make_thermal_overlay(img, ela_raw, alpha=0.55, blur_size=21)

    # Copy-Move
    cm = detect_copy_move(img, method=cm_algo, min_dist=cm_dist, ratio=cm_ratio)
    cm_overlay = make_thermal_overlay(img, cm["mask"], alpha=0.6, blur_size=31)

    # Noise
    noise_raw, noise_heat, noise_score = compute_noise_map(img, block_size=noise_win)
    noise_overlay = make_thermal_overlay(img, noise_raw, alpha=0.55, blur_size=25)

    # Combined
    fused_raw, fused_heat = fuse_evidence(ela_raw, cm["mask"], noise_raw)
    fused_overlay = make_thermal_overlay(img, fused_raw, alpha=0.6, blur_size=31)

    # Verdict
    result = get_verdict(
        ela_score, cm["confidence"], noise_score,
        cm["match_count"], cm["cluster_count"]
    )


# -- verdict banner ------------------------------------------------------------

st.markdown(
    f'<div class="verdict-box" style="background-color:{result["color"]}">'
    f'VERDICT: {result["verdict"]}'
    f'</div>',
    unsafe_allow_html=True
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Confidence", f"{result['risk_pct']}%")
m2.metric("Cloned Clusters", str(cm["cluster_count"]))
m3.metric("Keypoint Matches", str(cm["match_count"]))
m4.metric("Compression Gap", f"{int(ela_score * 100)}%")

st.markdown("---")


# -- result tabs ---------------------------------------------------------------

tabs = st.tabs(["Combined", "ELA", "Copy-Move", "Noise", "Report"])

# combined evidence
with tabs[0]:
    left, right = st.columns(2)
    with left:
        st.caption("Original")
        st.image(img, use_container_width=True)
    with right:
        st.caption("Tampering Heatmap")
        st.image(fused_overlay, use_container_width=True)

    st.caption("Standalone Heatmap")
    st.image(fused_heat, use_container_width=True)

    st.markdown("**Findings:**")
    for f in result["findings"]:
        st.write(f"- {f}")

# ELA tab
with tabs[1]:
    st.write(f"Quality: Q={ela_q}, Amplification: {ela_amp}x")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Original")
        st.image(img, use_container_width=True)
    with c2:
        st.caption("ELA Overlay")
        st.image(ela_overlay, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.caption("Amplified ELA")
        st.image(ela_amp_img, use_container_width=True)
    with c4:
        st.caption("ELA Heatmap")
        st.image(ela_heatmap, use_container_width=True)

# copy-move tab
with tabs[2]:
    st.write(f"Method: {cm_algo.upper()} | "
             f"Matches: {cm['match_count']} | "
             f"Clusters: {cm['cluster_count']}")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Match Vectors")
        st.image(cm["overlay"], use_container_width=True)
    with c2:
        st.caption("Thermal Heatmap")
        st.image(cm_overlay, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.caption("Detection Mask")
        st.image(cm["mask"], use_container_width=True)
    with c4:
        st.caption("Mask on Original")
        masked = draw_mask_overlay(img, cm["mask"])
        st.image(masked, use_container_width=True)

# noise tab
with tabs[3]:
    st.write(f"Window: {noise_win}x{noise_win}px")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Original")
        st.image(img, use_container_width=True)
    with c2:
        st.caption("Noise Overlay")
        st.image(noise_overlay, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.caption("Noise Heatmap")
        st.image(noise_heat, use_container_width=True)
    with c4:
        st.caption("Raw Deviation")
        st.image(noise_raw, use_container_width=True)

# report tab
with tabs[4]:
    report_text = f"""PIXELFORENSICS REPORT
{'='*50}
File: {fname}
Verdict: {result['verdict']}
Confidence: {result['risk_pct']}%

COPY-MOVE
  Matches: {cm['match_count']}
  Clusters: {cm['cluster_count']}
  Method: {cm_algo.upper()}
  Score: {cm['confidence']:.2f}

ELA
  Quality: Q={ela_q}
  Score: {ela_score:.2f}

NOISE
  Window: {noise_win}x{noise_win}
  Score: {noise_score:.2f}

FINDINGS
{chr(10).join('  - ' + x for x in result['findings'])}
{'='*50}
"""
    st.text_area("Report", report_text, height=300)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button("Download Report", report_text,
                           f"report_{fname}.txt", "text/plain")
    with d2:
        png = image_to_bytes(fused_overlay)
        st.download_button("Download Heatmap", png,
                           f"heatmap_{fname}.png", "image/png")

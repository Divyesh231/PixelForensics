"""
Utility script to generate realistic evaluation images for PixelForensics.
Produces:
  1. examples/sample_authentic.jpg - Untouched baseline photo
  2. examples/sample_copymove.jpg - Cloned object forgery
  3. examples/sample_spliced.jpg - Spliced element from another photo
"""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image

def generate_evaluation_suite():
    out_dir = Path("examples")
    out_dir.mkdir(exist_ok=True)

    # 1. Authentic baseline: Textured landscape / geometric subject
    h, w = 480, 680
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Background gradient (sky to ground)
    for y in range(h):
        r = int(100 + 80 * (y / h))
        g = int(130 + 60 * (y / h))
        b = int(180 - 40 * (y / h))
        canvas[y, :] = [b, g, r]

    # Foreground elements (distinct tree / geometric structures)
    cv2.circle(canvas, (180, 240), 50, (30, 160, 60), -1)
    cv2.rectangle(canvas, (165, 290), (195, 380), (40, 70, 120), -1)

    cv2.circle(canvas, (460, 260), 40, (20, 140, 200), -1)
    cv2.rectangle(canvas, (320, 310), (420, 410), (180, 80, 50), -1)

    # Add simulated camera sensor noise
    np.random.seed(42)
    sensor_noise = np.random.normal(0, 3.5, canvas.shape).astype(np.int16)
    authentic = np.clip(canvas.astype(np.int16) + sensor_noise, 0, 255).astype(np.uint8)

    p_authentic = out_dir / "sample_authentic.jpg"
    Image.fromarray(cv2.cvtColor(authentic, cv2.COLOR_BGR2RGB)).save(p_authentic, quality=95)
    print(f"Created: {p_authentic}")

    # 2. Copy-move tampered image (cloning the tree object)
    tampered_cm = authentic.copy()
    # Extract source patch (tree with foliage)
    patch = tampered_cm[180:370, 120:240].copy()
    # Paste clone at (x=480, y=180)
    tampered_cm[180:370, 460:580] = patch

    p_copymove = out_dir / "sample_copymove.jpg"
    Image.fromarray(cv2.cvtColor(tampered_cm, cv2.COLOR_BGR2RGB)).save(p_copymove, quality=92)
    print(f"Created: {p_copymove}")

    # 3. Spliced image (foreign element with mismatched compression & noise)
    tampered_spliced = authentic.copy()
    foreign_element = np.zeros((100, 100, 3), dtype=np.uint8)
    foreign_element[:] = (220, 20, 60)
    cv2.putText(foreign_element, "ALERT", (10, 55), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
    # Heavy foreign noise
    foreign_noise = np.random.normal(0, 15.0, foreign_element.shape).astype(np.int16)
    foreign_element = np.clip(foreign_element.astype(np.int16) + foreign_noise, 0, 255).astype(np.uint8)

    tampered_spliced[60:160, 280:380] = foreign_element
    p_spliced = out_dir / "sample_spliced.jpg"
    Image.fromarray(cv2.cvtColor(tampered_spliced, cv2.COLOR_BGR2RGB)).save(p_spliced, quality=88)
    print(f"Created: {p_spliced}")

if __name__ == "__main__":
    generate_evaluation_suite()

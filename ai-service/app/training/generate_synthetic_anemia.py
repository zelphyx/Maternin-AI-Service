"""
Generate synthetic conjunctival pallor image dataset for anemia training.

HONEST DOCUMENTATION:
- Real Indonesian conjunctival image dataset is not available locally.
- This script generates synthetic images that mimic conjunctiva color:
  - Normal conjunctiva: pink/red palette (RGB roughly 200-230, 150-180, 150-180)
  - Anemic conjunctiva: pale/white palette (RGB roughly 220-250, 200-220, 200-220)
- Labels are deterministic from the color profile — no real clinical validation.
- This is documented as a synthetic baseline. Real deployment would need
  labeled Indonesian patient images with confirmed Hb values.

Output:
    datasets/anemia_synthetic/
      train/
        anemia/  (200 images)
        normal/  (200 images)
      test/
        anemia/  (50 images)
        normal/  (50 images)
"""
import os
import numpy as np
from PIL import Image

np.random.seed(42)

OUTPUT_DIR = "/Users/zelphyx/Projects/Maternin-AI/datasets/anemia_synthetic"
IMG_SIZE = 224
N_TRAIN_PER_CLASS = 200
N_TEST_PER_CLASS = 50


def make_conjunctiva_image(pale: bool, seed: int) -> Image.Image:
    """Generate synthetic 224x224 RGB image mimicking conjunctiva.

    Conjunctiva (inner eyelid) characteristics:
    - Background skin tone around eye (periorbital area)
    - Reddish/pinkish area showing blood vessels (the conjunctiva)
    - Pale conjunctiva: less red, more white/pink

    Args:
        pale: True for anemic (pale), False for normal (pink/red)
        seed: random seed
    """
    rng = np.random.RandomState(seed)
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    # Base skin color (periorbital area) — slightly dark for Indonesian population
    skin_r, skin_g, skin_b = 130 + rng.randint(-15, 15), 90 + rng.randint(-10, 10), 70 + rng.randint(-10, 10)
    img[:, :] = [skin_r, skin_g, skin_b]

    # Conjunctiva ROI — central area
    cy, cx = IMG_SIZE // 2, IMG_SIZE // 2
    y_grid, x_grid = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    dist = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)
    in_conjunctiva = dist < 60

    if pale:
        # Pale conjunctiva: very light, washed out, low R channel
        # RGB roughly (220-250, 200-225, 200-225) — almost white/pink
        cj_r = 220 + rng.randint(0, 30)
        cj_g = 200 + rng.randint(0, 25)
        cj_b = 200 + rng.randint(0, 25)
    else:
        # Normal conjunctiva: pink/red — high R, lower G/B
        # RGB roughly (195-225, 130-160, 130-160) — visibly red
        cj_r = 195 + rng.randint(0, 30)
        cj_g = 130 + rng.randint(0, 30)
        cj_b = 130 + rng.randint(0, 30)

    img[in_conjunctiva] = [cj_r, cj_g, cj_b]

    # Add some noise
    noise = rng.randint(-15, 15, img.shape, dtype=np.int8)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add a few dark blood vessel lines (for normal conjunctiva, more prominent)
    if not pale:
        n_vessels = rng.randint(5, 12)
        for _ in range(n_vessels):
            vx = cx + rng.randint(-40, 40)
            vy = cy + rng.randint(-40, 40)
            length = rng.randint(10, 30)
            for dx in range(-length // 2, length // 2):
                if 0 <= vx + dx < IMG_SIZE and 0 <= vy < IMG_SIZE:
                    img[vy, vx + dx] = [150, 50, 50]

    return Image.fromarray(img)


def generate_split(split: str, n_per_class: int) -> None:
    split_dir = os.path.join(OUTPUT_DIR, split)
    for label in ["anemia", "normal"]:
        label_dir = os.path.join(split_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        is_pale = (label == "anemia")
        base_seed = 0 if split == "train" else 10000
        for i in range(n_per_class):
            img = make_conjunctiva_image(pale=is_pale, seed=base_seed + i)
            path = os.path.join(label_dir, f"{label}_{i:04d}.jpg")
            img.save(path, "JPEG", quality=90)
    print(f"  {split}: {n_per_class} anemia + {n_per_class} normal = {2 * n_per_class} images")


def main():
    print("=" * 60)
    print("Synthetic Anemia Conjunctival Dataset Generator")
    print("=" * 60)
    print("\nNOTE: This generates SYNTHETIC images for training baseline.")
    print("Real deployment requires labeled Indonesian patient data with Hb.")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating training set...")
    generate_split("train", N_TRAIN_PER_CLASS)
    print("Generating test set...")
    generate_split("test", N_TEST_PER_CLASS)

    print()
    print(f"Dataset ready at: {OUTPUT_DIR}")
    print("Total: 500 images (400 train + 100 test)")


if __name__ == "__main__":
    main()

import cv2
import os
import glob

# === Configuration ===
input_dir = 'img'                 # Folder with images + labels
output_dir = 'cropped_hands'     # Output folder

os.makedirs(output_dir, exist_ok=True)

# === Label parser function ===
def parse_label(label_path, img_w, img_h):
    with open(label_path, 'r') as f:
        parts = f.readline().strip().split()
        values = list(map(float, parts[1:]))  # skip class_id

    if len(values) == 8:
        # Polygon format (4 points)
        points = [(int(values[i] * img_w), int(values[i + 1] * img_h)) for i in range(0, 8, 2)]
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        x_min, x_max = max(min(x_vals), 0), min(max(x_vals), img_w)
        y_min, y_max = max(min(y_vals), 0), min(max(y_vals), img_h)

    elif len(values) == 4:
        # YOLO bbox format: x_center, y_center, width, height
        x_center, y_center, bw, bh = values
        x_min = int((x_center - bw / 2) * img_w)
        y_min = int((y_center - bh / 2) * img_h)
        x_max = int((x_center + bw / 2) * img_w)
        y_max = int((y_center + bh / 2) * img_h)
        # Clamp to image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_w, x_max)
        y_max = min(img_h, y_max)

    else:
        raise ValueError(f"Unexpected label format in {label_path} ({len(values)} coords)")

    return x_min, x_max, y_min, y_max

# === Process all .jpg files ===
image_paths = sorted(glob.glob(os.path.join(input_dir, '*.jpg')))

for image_path in image_paths:
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(input_dir, base_name + '.txt')

    if not os.path.exists(label_path):
        print(f"[!] Missing label for {base_name}, skipping.")
        continue

    img = cv2.imread(image_path)
    if img is None:
        print(f"[!] Cannot read {image_path}, skipping.")
        continue
    h, w = img.shape[:2]

    try:
        x_min, x_max, y_min, y_max = parse_label(label_path, w, h)
        cropped = img[y_min:y_max, x_min:x_max]
        out_path = os.path.join(output_dir, base_name + '.jpg')
        cv2.imwrite(out_path, cropped)
        print(f"[✓] Saved: {out_path}")
    except Exception as e:
        print(f"[✗] Error with {base_name}: {e}")

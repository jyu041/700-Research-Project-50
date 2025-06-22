import cv2
import os
import glob
import sys

# === Config ===
image_dir = 'img'
output_img_dir = 'labeled_images'
output_lbl_dir = 'labels_yolo'
class_id = 0

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_lbl_dir, exist_ok=True)

image_paths = sorted(
    glob.glob(os.path.join(image_dir, '*.jpg')) +
    glob.glob(os.path.join(image_dir, '*.png'))
)

def save_yolo_bbox(image_shape, bbox, label_path):
    ih, iw = image_shape[:2]
    x, y, w, h = bbox
    xc = (x + w / 2) / iw
    yc = (y + h / 2) / ih
    wn = w / iw
    hn = h / ih
    with open(label_path, 'w') as f:
        f.write(f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

def label_images():
    print("Instructions:\n"
          "  • Draw box → Enter/Space to finish\n"
          "  • After drawing: y = accept, r = redraw, s = skip, q = quit\n")

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        label_path = os.path.join(output_lbl_dir, os.path.splitext(filename)[0] + '.txt')

        img = cv2.imread(img_path)
        if img is None:
            print(f"[!] Could not open {img_path}")
            continue

        while True:
            # Draw ROI box
            roi = cv2.selectROI("Draw box and press Enter", img, fromCenter=False, showCrosshair=True)

            if roi == (0, 0, 0, 0):
                print("[!] No ROI selected.")
                key = input("Press [s] to skip, [q] to quit, any other key to retry: ")
                if key.lower() == 's':
                    break
                elif key.lower() == 'q':
                    cv2.destroyAllWindows()
                    sys.exit(0)
                continue

            # Show confirmation window
            preview = img.copy()
            x, y, w, h = roi
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Confirm (y = save, r = redraw, s = skip, q = quit)", preview)

            key = cv2.waitKey(0) & 0xFF
            cv2.destroyWindow("Confirm (y = save, r = redraw, s = skip, q = quit)")

            if key == ord('y'):
                save_yolo_bbox(img.shape, roi, label_path)
                cv2.imwrite(os.path.join(output_img_dir, filename), img)
                print(f"[✓] Saved label for {filename}")
                break
            elif key == ord('r'):
                continue  # redraw
            elif key == ord('s'):
                print("[→] Skipped")
                break
            elif key == ord('q'):
                print("[x] Quit")
                cv2.destroyAllWindows()
                sys.exit(0)

    cv2.destroyAllWindows()
    print("Finished labeling.")

if __name__ == "__main__":
    label_images()

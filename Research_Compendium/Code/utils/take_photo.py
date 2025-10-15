import cv2
import os
import time

# === Config ===
base_name = "Q"
output_folder = "captures"
start_index = 0
capture_delay = 0

os.makedirs(output_folder, exist_ok=True)

current_index = start_index
while os.path.exists(os.path.join(output_folder, f"{base_name}_{current_index}.jpg")):
    current_index += 1

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

print("📸 Press 's' to start delayed capture, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        break

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        print(f"⏳ Taking photo in {capture_delay} seconds...")
        time.sleep(capture_delay)

        ret, frame = cap.read()
        filename = f"{base_name}_{current_index}.jpg"
        filepath = os.path.join(output_folder, filename)
        cv2.imwrite(filepath, frame)
        print(f"✅ Saved: {filepath}")
        current_index += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

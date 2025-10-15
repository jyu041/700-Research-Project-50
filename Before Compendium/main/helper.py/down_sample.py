import os
import shutil

# === Configuration ===
base_input_dir = r'C:\Users\Tony\Downloads\700-Research-Project-50\dataset\set_1\asl_alphabet_train'
base_output_dir = r'C:\Users\Tony\Downloads\asl_sampled'
sample_interval = 3
offset = 0

# === Create base output directory ===
os.makedirs(base_output_dir, exist_ok=True)

# === Iterate through all subfolders (A-Z, del, space, nothing) ===
for label_folder in os.listdir(base_input_dir):
    label_path = os.path.join(base_input_dir, label_folder)
    if not os.path.isdir(label_path):
        continue  # skip non-folder entries

    output_label_path = os.path.join(base_output_dir, label_folder)
    os.makedirs(output_label_path, exist_ok=True)

    # Get and filter image files
    all_files = sorted(os.listdir(label_path))
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Sample and copy
    for i in range(offset, len(image_files), sample_interval):
        src = os.path.join(label_path, image_files[i])
        dst = os.path.join(output_label_path, image_files[i])
        shutil.copy2(src, dst)
        print(f"Copied {image_files[i]} from {label_folder}")

print("All sampling complete.")

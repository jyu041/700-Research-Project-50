import os

# === Config ===
labels_dir = 'labels_yolo'          # Folder with original .txt label files
output_dir = 'labels_remapped'      # Folder to save updated label files
os.makedirs(output_dir, exist_ok=True)

# === Class name → ID mapping based on your data.yaml
custom_class_map = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3,
    'DEL': 4,
    'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9,
    'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14,
    'NOTHING': 15,
    'O': 16, 'P': 17, 'Q': 18, 'R': 19, 'S': 20,
    'SPACE': 21,
    'T': 22, 'U': 23, 'V': 24, 'W': 25, 'X': 26, 'Y': 27, 'Z': 28
}

# === Helper: infer class name from filename
def get_class_name_from_filename(filename):
    upper = filename.upper()
    if upper.startswith("DEL"):
        return "DEL"
    elif upper.startswith("NOTHING"):
        return "NOTHING"
    elif upper.startswith("SPACE"):
        return "SPACE"
    else:
        return upper[0] if upper[0].isalpha() else None

# === Remap label files
for file in os.listdir(labels_dir):
    if not file.endswith('.txt'):
        continue

    class_name = get_class_name_from_filename(file)
    if class_name not in custom_class_map:
        print(f"Skipping {file} — unknown class prefix '{class_name}'")
        continue

    new_class_id = custom_class_map[class_name]
    input_path = os.path.join(labels_dir, file)
    output_path = os.path.join(output_dir, file)

    with open(input_path, 'r') as fin:
        lines = fin.readlines()

    with open(output_path, 'w') as fout:
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                parts[0] = str(new_class_id)
                fout.write(' '.join(parts) + '\n')

    print(f"✔ Updated {file}: class '{class_name}' → {new_class_id}")

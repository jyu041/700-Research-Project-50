# make_labels.py
import os, json

data_root = r"C:\Users\Tony\Downloads\compsys700\700-Research-Project-50\dataset\all"

classes = [d for d in os.listdir(data_root)
           if os.path.isdir(os.path.join(data_root, d))]
classes = sorted(classes)  # EXACTLY what training used

print("Class count:", len(classes))
print("First 10:", classes[:10])

with open("labels.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(classes))

with open("labels.json", "w", encoding="utf-8") as f:
    json.dump(classes, f, ensure_ascii=False, indent=2)

print("Wrote labels.txt and labels.json")

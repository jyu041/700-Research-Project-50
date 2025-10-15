import json

# === Load the JSON file ===
with open("summary_ASL+BSL_tflite.json", "r", encoding="utf-8") as f:
    data = json.load(f)

classes = data["classes"]
accuracies = data["per_class_accuracy"]

# === Match class with its accuracy ===
class_accuracy = {cls: acc for cls, acc in zip(classes, accuracies)}

# === Sort alphabetically and print ===
for cls in sorted(class_accuracy.keys()):
    print(f"{cls}: {class_accuracy[cls]:.3f}")

# (Optional) Save to CSV
import csv
with open("class_accuracy.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Class", "Accuracy"])
    for cls, acc in sorted(class_accuracy.items()):
        writer.writerow([cls, acc])

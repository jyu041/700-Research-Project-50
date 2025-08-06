import tensorflow as tf

# Load the SavedModel
converter = tf.lite.TFLiteConverter.from_saved_model("best_hand_tf")

# (Optional) Enable optimizations for smaller/faster models
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save it to disk
with open("best_hand.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved as best_hand.tflite")

import tensorflow as tf

saved_model_dir = "out_tf"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # enables quantization & graph optimizations
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved as model.tflite")

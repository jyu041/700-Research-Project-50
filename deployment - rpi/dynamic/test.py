import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
print("✅ Model loaded. Input details:", interpreter.get_input_details())
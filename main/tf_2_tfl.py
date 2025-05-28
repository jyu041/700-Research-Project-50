import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('model.h5')

# Convert to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save to .tflite file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

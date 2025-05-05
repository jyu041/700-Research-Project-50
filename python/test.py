import tensorflow as tf

# Path to dataset
data_dir = "../dataset/set_1/asl_alphabet_train"

# Parameters
batch_size = 32
img_height = 64
img_width = 64
seed = 123  # for shuffling consistency

# Load dataset (automatically labels by folder names)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,      # Split 20% for validation
    subset="training",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,      # Same 20% split
    subset="validation",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Class names (folder names)
class_names = train_ds.class_names
print("Class names:", class_names)

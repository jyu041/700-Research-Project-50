# Project #50: Sign Language Recognition on a Single Board Computer

## Overview

[Sign Language Recognition on a Single Board Computer](https://part4project.foe.auckland.ac.nz/home/project/detail/5580/)

In this project, we developed and evaluated a system designed to recognise as many hand signs as possible in real-time and as accurately as possible. Our goal was to determine what is practically possible on a lightweight system such as a Raspberry Pi.

## Supervisors

* [Trevor Gee](https://profiles.auckland.ac.nz/t-gee)
* [Ho Seok Ahn](https://profiles.auckland.ac.nz/hs-ahn)

## Team Members

* [Jerry Yu](https://github.com/jyu041)
* [Tony Huang](https://github.com/H-qitai)

---

## Project Story & Documentation

To provide a clear view of how our project evolved, we included the following supporting documents in the `Story/` folder:

- **ExpectedTimeFrames.pdf** – Contains the expected timelines we initially set for ourselves, outlining planned milestones and deliverables.
- **RealTimeFrames.pdf** – Records the actual timeline we followed, highlighting deviations from our original plan.
- **TimeFramesComparison.pdf** – A short comparative report between our expected and real timelines, offering insight into project management challenges and adjustments.
- **MeetingLogs.pdf** – Includes brief summaries of meeting sessions and notes, showing our decision-making process throughout the project.
- **OriginalTimeLine.png** – A Gantt chart image of our original expected timeline for quick visual reference.

These documents help track the evolution of our project, comparing expectations with reality, and providing transparency in how we collaborated and managed progress.

---

## Datasets

### Dataset Structure

- **ASL**: Dataset used to train ASL-only models
- **ASL+BSL**: Dataset used to train combined ASL+BSL models
- **BSL**: Separate folder that can be used to train a BSL-only model
- **YOLO_classify**: Failed attempt of YOLO tracking and classification dataset
- **YOLO_hand_only**: Dataset used for training ASL hand tracking
- **YOLO_hand_only_ASL+BSL**: Dataset used for training ASL+BSL hand tracking
- **test**: Images used for testing model accuracy

### Full Dataset Downloads

**Note**: Datasets are too large for this compendium. Full datasets can be downloaded from the following sources:

- **Full ASL**: [Kaggle - ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Full BSL**: [Kaggle - BSL Static Alphabet](https://www.kaggle.com/datasets/philipilono/bsl-static-alphabet)
- **WLASL**: [Kaggle - WLASL Processed](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed/data?select=videos)
- **LSA64**: [Official LSA64 Dataset](https://facundoq.github.io/datasets/lsa64/)

---

## Environment Setup

### Workstation A

**Hardware Specifications:**
- GPU: NVIDIA RTX 3060 (12 GB VRAM)
- RAM: 16 GB
- CPU: Intel i5 11th Generation
- OS: Windows

**Environment Setup:**

The `Workstation_A.yml` file can be used to recreate the software environment from Anaconda for Workstation A.

```bash
conda env create -f Workstation_A.yml
```

### Workstation B

**Hardware Specifications:**
- GPU: NVIDIA RTX 3070 (8 GB VRAM)
- RAM: 16 GB
- CPU: AMD Ryzen 7 5800X
- OS: Windows

**Environment Setup:**

The `Workstation_B.yml` file can be used to recreate the software environment from Anaconda for Workstation B.

```bash
conda env create -f Workstation_B.yml
```

### Raspberry Pi 5

**Hardware Specifications:**
- RAM: 16 GB
- CPU: Arm Cortex A76 @ 2.4 GHz
- Camera: External Logitech C922

---

## Code Structure

### Evaluation

#### Test Scripts Overview

- **`test_classifier.py`**  
  Tests the MobileNetV2 model.  
  Ensure `--model` points to the correct TensorFlow/TensorFlow Lite model and `--data_dir` points to the right `test_images` folder.

- **`test_combined_v5.py`**  
  Tests the two-stage pipeline tailored for Raspberry Pi 5.  
  Ensure both models point to the right path and test image.

- **`test_combined_v8.py`**  
  Tests the two-stage pipeline from Workstation A.  
  Ensure both models point to the right path and test image.

- **`test_dynamic_pt.py`**  
  Tests the dynamic model with PyTorch framework.  
  Ensure model points to the right path and test video.

- **`test_dynamic_tflite.py`**  
  Tests the TensorFlow Lite model with TFLite framework.  
  Ensure model points to the right path and test video.

### Inference

#### Available Interpreter Scripts

- **`asl_intepreter_YOLOv8n.py`**  
  Version transformed directly from Workstation A. Does not perform well in terms of resource usage on Raspberry Pi 5.

- **`asl_intepreter_YOLOv5n.py`**  
  Final version tailored for Raspberry Pi 5 (recommended).

- **`dynamic_interpreter.py`**  
  TensorFlow version of the dynamic interpreter.

#### Configuration Requirements

**For `asl_intepreter_YOLOv8n.py`:**
- `--model_path` → Point to the correct classifier
- `--yolo_weights` → Point to the correct YOLO model

**For `asl_intepreter_YOLOv5n.py`:**
- `--det_tflite` → Point to the correct detector model
- `--cls_tflite` → Point to the correct classifier model

**For Both Scripts:**
- `--labels` → Use `labels_classifier.json` if running ASL+BSL

**Script Modifications Required:**
- **Line 50**: Point to `lsa64_fast_20250917_203048_meta.tflite` model
- **Line 51**: Change label to `lsa64_fast_20250917_203048_meta.json`

**Note**: All models can be found inside the `Model` folder.

### Models

#### Model-Label Mappings

| Model Files | Label File | Notes |
|------------|------------|-------|
| `model_ASL+BSL.h5`<br>`model_ASL+BSL.tflite` | `labels_classifier.json` | ASL + BSL combined model |
| `model_ASL.h5`<br>`model_ASL.tflite` | None needed | ASL-only model |
| `lsa64_fast_20250917_203048_meta.pt`<br>`lsa64_fast_20250917_203048_meta.tflite` | `lsa64_fast_20250917_203048_meta.json` | LSA64 model |

#### YOLO Models

- **`v5n_100epoch_img416.tflite`** - YOLOv5n model
- **`v8n_100epoch_img416.pt`** - YOLOv8n model
- **`YOLO_classify/`** - Contains failed YOLO tracking and classification models

### Training

#### MobileNetV2

Training code is located in the `MobileNetV2` directory.  
Ensure `--data_root` points to the correct folder path when training.

#### YOLOv8n

Training is done by:
1. Installing Ultralytics
2. Running the command:
   ```bash
   yolo task=detect mode=train model=yolov8n.pt data=dataset/data.yaml epochs=100 imgsz=416 device=0
   ```
   Ensure `dataset` points to the right path.

#### LSA64

Training code is in the `LSA64` directory.

**No input arguments are needed.** Modifications can be made directly in the file:
- **Line 57**: Modify the dataset directory
- **Top of file**: Hyperparameters can be tuned, but it is recommended to keep them unmodified

**Steps:**
1. Create a `models` folder in the same directory to save trained models
2. Run `python train.py`

#### WLASL

Contains a Python notebook for training the LSTM model on the WLASL dataset.

**Configuration:**
- **First cell**: Dataset directory can be modified in the `Config` class
- **Config class**: Hyperparameters can be tuned, but it is recommended to keep them unmodified

**Steps:**
1. Ensure the dataset is downloaded
2. Run all notebook cells in order

### Utilities

#### `down_sample.py`

Samples images from the ASL alphabet dataset to create a smaller subset for training or testing.

**Features:**
- Loops through all class folders (A–Z, del, space, nothing)
- Copies every Nth image (set by `sample_interval`)
- Maintains original folder structure
- Only copies `.png`, `.jpg`, and `.jpeg` files
- Preserves file metadata

#### `labeller_yolo.py`

Manual labeling interface for images to generate YOLO-format `.txt` annotation files.

**Usage:**
1. Opens each image for bounding box drawing
2. Press Enter/Space to finish drawing
3. **Keyboard Controls:**
   - `y` = Save box
   - `r` = Redraw
   - `s` = Skip
   - `q` = Quit

**Output:**
- Labels saved in `labels_yolo/`
- Labeled images saved in `labeled_images/`
- Bounding boxes are normalized for YOLO training

#### `take_photo.py`

Captures labeled photos directly from a webcam.

**Controls:**
- `s` = Take photo (with optional delay)
- `q` = Quit

**Features:**
- Files saved as `<base_name>_<index>.jpg` in `captures/` folder
- Automatically continues indexing from the last saved file

#### `hand_extract_for_yolo.py`

Crops hands from images based on YOLO label files.

**Features:**
- Reads images and corresponding `.txt` labels
- Supports YOLO bounding boxes and polygon annotations
- Crops hand regions and saves to `cropped_hands/`
- Automatically skips missing or unreadable files

---

## Results

### Model Evaluation Directory Structure

#### ASL Dataset

- **ASL**: Contains only ASL signs and evaluation data
- **eval_results_CNN**: Evaluation results from MobileNetV2 classifier only
- **eval_results_two_stage**: Evaluation results from YOLO + MobileNetV2 pipeline
- **eval_tf**: TensorFlow model and PyTorch models (MobileNetV2 and YOLO)
- **eval_tflite**: TensorFlow Lite converted versions of YOLO and MobileNetV2

#### ASL+BSL Dataset

Follows the same structure as the ASL dataset.

#### LSA64 Dataset

- **LSA64**: Contains evaluation results for dynamic hand signs
- **eval_pytorch**: Model evaluation using the PyTorch model
- **eval_tflite**: Model evaluation using the converted TFLite model

---
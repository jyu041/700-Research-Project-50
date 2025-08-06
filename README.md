# Project #50: 
## [Sign Language Recognition on a Single Board Computer](https://part4project.foe.auckland.ac.nz/home/project/detail/5580/)
In this project, we will develop and evaluate a system designed to recognise as many hand signs as possible in real-time and as accurately as possible. Our goal is to determine what is practically possible on a lightweight system such as a Raspberry Pi.

## Supervisors:
* [Trevor Gee](https://profiles.auckland.ac.nz/t-gee)
* [Ho Seok Ahn](https://profiles.auckland.ac.nz/hs-ahn)

## Team Members:
* [Jerry Yu](https://github.com/jyu041)
* [Tony Huang](https://github.com/H-qitai)

## Design
[Here](./Design.md) you can find our proposed design.

## Datasets
[Here](./Datasets.md) shows a list of datasets that we can choose to use.

## .pt to .tflite
```bash
pip install ultralytics onnx onnx-tf
pip install tensorflow==2.14.0
pip install tensorflow-probability==0.22.0
pip install numpy==1.24.4
```

```bash
yolo export model=best_hand.pt format=onnx opset=12 dynamic=False
```

```bash
onnx-tf convert -i best_hand.onnx -o best_hand_tf
```
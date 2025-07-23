from ultralytics import YOLO

model = YOLO('best_hand.pt')

model.export(format='tflite', imgsz=640, int8=True)
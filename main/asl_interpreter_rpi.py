#!/usr/bin/env python3
import argparse
import numpy as np
import tflite_runtime.interpreter as tflite
from datetime import datetime
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
from picamera2 import Picamera2

# Define ASL alphabet labels
ASL_LABELS = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del','nothing','space'
]


def parse_arguments():
    parser = argparse.ArgumentParser(description='ASL Sign Interpreter on Raspberry Pi without OpenCV')
    parser.add_argument('--model_path', type=str, default='./model.tflite',
                        help='Path to the TFLite model file (.tflite)')
    parser.add_argument('--confidence', type=float, default=0.7,
                        help='Confidence threshold for predictions (default: 0.7)')
    parser.add_argument('--record', action='store_true',
                        help='Save annotated frames to disk instead of displaying')
    return parser.parse_args()


def load_model(path):
    print(f"Loading TFLite model from {path}...")
    interpreter = tflite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    _, h, w, _ = inp['shape']
    return interpreter, inp['index'], out['index'], (w, h), inp['dtype']


def preprocess(pil_img, target_size):
    # center-crop
    w, h = pil_img.size
    s = min(w, h)
    left, top = (w - s)//2, (h - s)//2
    cropped = pil_img.crop((left, top, left+s, top+s)).resize(target_size)
    return np.asarray(cropped, dtype=np.float32) / 255.0


def annotate(pil_img, label, conf, threshold):
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()
    text = f"{label} ({conf:.2f})" if conf >= threshold else "Uncertain"
    color = (0,255,0) if conf >= threshold else (255,0,0)
    draw.text((10,10), text, fill=color, font=font)
    return pil_img


def main():
    args = parse_arguments()
    interpreter, inp_idx, out_idx, input_size, inp_dtype = load_model(args.model_path)

    # Prepare PiCamera2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration({"size": (640,480)})
    picam2.configure(config)
    picam2.start()

    frame_count = 0
    if args.record:
        os.makedirs('annotated_frames', exist_ok=True)

    print("Starting ASL interpretation. Press Ctrl+C to stop.")
    try:
        while True:
            arr = picam2.capture_array()  # RGB array
            pil_img = Image.fromarray(arr)

            proc = preprocess(pil_img, input_size)
            inp_tensor = np.expand_dims(proc, 0).astype(inp_dtype)
            interpreter.set_tensor(inp_idx, inp_tensor)
            interpreter.invoke()
            preds = interpreter.get_tensor(out_idx)[0]
            idx = int(np.argmax(preds))
            conf = float(preds[idx])
            label = ASL_LABELS[idx]

            # Annotate
            out_img = annotate(pil_img.copy(), label, conf, args.confidence)

            if args.record:
                fname = f"annotated_frames/frame_{frame_count:05d}.jpg"
                out_img.save(fname)
                frame_count += 1
            else:
                cv2.imshow("ASL Interpreter",
                           cv2.cvtColor(np.array(out_img), cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == 27:   # Esc key quits the loop
                    break
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        picam2.close()
        print("Interpreter closed.")

if __name__ == '__main__':
    main()

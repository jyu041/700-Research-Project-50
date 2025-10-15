# export_onnx.py
import torch, json
from pathlib import Path
from typing import List
from your_module_with_classes import VideoTransformerClassifier, build_model_from_ckpt, load_meta_for_model  # adjust import

MODEL_PATH = "dynamic.pt"
META_JSON_PATH = "dynamic.json"   # or None
NUM_FRAMES = 8
FRAME_SIZE = 112

def main():
    device = torch.device("cpu")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    meta = load_meta_for_model(MODEL_PATH, META_JSON_PATH)
    classes: List[str] = meta.get("classes", ckpt.get("classes", []))
    assert classes, "No classes found."

    model = build_model_from_ckpt(ckpt, num_classes=len(classes)).eval()
    # Make sure we’re not using channels_last, AMP, etc.
    model = model.to(device)

    # Example input: [1, T, 3, H, W]
    x = torch.zeros(1, NUM_FRAMES, 3, FRAME_SIZE, FRAME_SIZE, dtype=torch.float32, device=device)

    onnx_path = "model_fp32_opset17.onnx"
    torch.onnx.export(
        model, x, onnx_path,
        export_params=True,
        opset_version=17,          # good balance of features
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None          # keep static
    )
    print("Saved:", onnx_path)

if __name__ == "__main__":
    main()

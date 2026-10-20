
import torch
import os

models_dir = os.path.expanduser("~/darts_edge/models")
pth_path = os.path.join(models_dir, "gray_trajLen_3_model.pth")
onnx_path = os.path.join(models_dir, "cnn_classifier.onnx")
engine_path = os.path.join(models_dir, "cnn_classifier.engine")

# Load the TorchScript model
print("Loading PyTorch model...")
model = torch.jit.load(pth_path, map_location="cpu")
model.eval()

# Export to ONNX
print("Exporting to ONNX...")
dummy = torch.randn(1, 3, 56, 56)
torch.onnx.export(model, dummy, onnx_path,
                  input_names=["input"], output_names=["output"],
                  opset_version=11)
print(f"ONNX saved: {onnx_path} ({os.path.getsize(onnx_path)} bytes)")

# Build TRT engine using trtexec
import subprocess
trtexec = "/usr/src/tensorrt/bin/trtexec"
if not os.path.exists(trtexec):
    # Try alternative location
    import shutil
    trtexec = shutil.which("trtexec") or trtexec

cmd = [trtexec, f"--onnx={onnx_path}", f"--saveEngine={engine_path}", "--fp16", "--workspace=1024"]
print(f"Building TRT engine: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
if result.returncode == 0:
    print(f"TRT engine built: {engine_path} ({os.path.getsize(engine_path)} bytes)")
else:
    print(f"trtexec failed (rc={result.returncode})")
    print(result.stderr[-500:] if result.stderr else "no stderr")

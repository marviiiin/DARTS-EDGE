
import numpy as np
import json
import os
import sys
import torch

# Load FP32 model first, get predictions, then unload
print("Loading PyTorch FP32 model...")
model = torch.jit.load("models/gray_trajLen_3_model.pth", map_location="cpu")
model.eval()

# Generate 100 random test images (simulate trajectory images)
np.random.seed(42)
n_test = 200
test_images = np.random.randint(0, 256, (n_test, 56, 56, 3), dtype=np.uint8)

# Also create some structured patterns
# Black image with white lines (like actual trajectory images)
for i in range(50):
    img = np.zeros((56, 56, 3), dtype=np.uint8)
    # Draw some random white lines
    n_lines = np.random.randint(2, 8)
    for _ in range(n_lines):
        x_start = np.random.randint(0, 56)
        y_start = np.random.randint(0, 56)
        x_end = np.random.randint(0, 56)
        y_end = np.random.randint(0, 56)
        # Draw a simple line
        pts = max(abs(x_end - x_start), abs(y_end - y_start), 1)
        for t in range(pts):
            x = int(x_start + (x_end - x_start) * t / pts)
            y = int(y_start + (y_end - y_start) * t / pts)
            if 0 <= x < 56 and 0 <= y < 56:
                img[y, x] = [255, 255, 255]
    test_images[150 + i] = img

# FP32 predictions
batch = np.zeros((n_test, 3, 56, 56), dtype=np.float32)
for i in range(n_test):
    batch[i] = (test_images[i].astype(np.float32) / 255.0).transpose(2, 0, 1)

tensor = torch.from_numpy(batch)
with torch.no_grad():
    out = model(tensor)
    fp32_preds = torch.max(out, 1)[1].cpu().numpy()
    fp32_probs = torch.softmax(out, 1).cpu().numpy()

print("FP32 predictions done: %d images" % n_test)
fp32_dist = {0: int(np.sum(fp32_preds == 0)), 1: int(np.sum(fp32_preds == 1)), 2: int(np.sum(fp32_preds == 2))}
print("FP32 distribution: %s" % fp32_dist)

# Unload PyTorch model
del model, tensor, out
torch.cuda.empty_cache() if torch.cuda.is_available() else None
import gc; gc.collect()

# Now load TRT engine
print("\nLoading TensorRT FP16 engine...")
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    cuda.init()
    ctx = cuda.Device(0).make_context()

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    with open("models/cnn_classifier.engine", "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # Process in batches of 16
    fp16_preds = np.zeros(n_test, dtype=np.int32)
    batch_size = 16

    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        bs = end - start
        context.set_input_shape("input", (bs, 3, 56, 56))

        h_input = batch[start:end].copy()
        h_output = np.empty((bs, 3), dtype=np.float32)

        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)

        cuda.memcpy_htod(d_input, h_input)
        context.set_tensor_address("input", int(d_input))
        context.set_tensor_address("output", int(d_output))
        context.execute_async_v3(stream_handle=0)
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(h_output, d_output)

        fp16_preds[start:end] = np.argmax(h_output, axis=1)

        d_input.free()
        d_output.free()

    print("FP16 predictions done: %d images" % n_test)
    fp16_dist = {0: int(np.sum(fp16_preds == 0)), 1: int(np.sum(fp16_preds == 1)), 2: int(np.sum(fp16_preds == 2))}
    print("FP16 distribution: %s" % fp16_dist)

    # Compare
    agree = int(np.sum(fp32_preds == fp16_preds))
    print("\nAgreement: %d/%d (%.2f%%)" % (agree, n_test, agree/n_test*100))

    # Show disagreements
    disagree_idx = np.where(fp32_preds != fp16_preds)[0]
    class_names = ["Normal", "Congestion", "Incident"]
    if len(disagree_idx) > 0:
        print("Disagreements:")
        for idx in disagree_idx[:10]:
            print("  img %d: FP32=%s FP16=%s  FP32_probs=%s" % (
                idx, class_names[fp32_preds[idx]], class_names[fp16_preds[idx]],
                fp32_probs[idx].round(4)))

    # Save results
    results = {
        "n_images": n_test,
        "agreement": agree,
        "agreement_pct": round(agree / n_test * 100, 2),
        "fp32_distribution": fp32_dist,
        "fp16_distribution": fp16_dist,
        "n_disagreements": len(disagree_idx),
    }
    with open("tcdnet_fp16_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved: tcdnet_fp16_comparison.json")
    print(json.dumps(results, indent=2))

    ctx.pop()
except Exception as e:
    print("TRT error: %s" % e)
    import traceback; traceback.print_exc()

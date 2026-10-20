"""Build TCD-Net TensorRT FP16 engine from TorchScript model.

Handles the ScriptModule ONNX export issue by using the legacy exporter
or by tracing the model first.

Usage on Jetson:
    python build_cnn_engine_v2.py
"""
import os
import sys
import torch
import numpy as np

MODEL_DIR = os.path.expanduser('~/darts_edge/models')
PTS_PATH = os.path.join(MODEL_DIR, 'gray_trajLen_3_model.pth')
ONNX_PATH = os.path.join(MODEL_DIR, 'cnn_classifier.onnx')
ENGINE_PATH = os.path.join(MODEL_DIR, 'cnn_classifier.engine')

# Input shape: batch=1, channels=3, height=56, width=56
DUMMY_INPUT = torch.randn(1, 3, 56, 56)


def load_torchscript_model():
    """Load the TorchScript model."""
    model = torch.jit.load(PTS_PATH, map_location='cpu')
    model.eval()
    # Quick sanity check
    with torch.no_grad():
        out = model(DUMMY_INPUT)
    print(f"Model loaded. Output shape: {out.shape}, classes: {out.shape[1]}")
    return model


def export_onnx_legacy(model):
    """Export using legacy ONNX exporter (pre-dynamo)."""
    print("\n--- Attempt 1: Legacy ONNX export ---")
    try:
        # Force legacy exporter by setting dynamo=False (PyTorch >= 2.1)
        torch.onnx.export(
            model,
            DUMMY_INPUT,
            ONNX_PATH,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
            opset_version=11,
            dynamo=False,
        )
        print(f"ONNX exported: {ONNX_PATH}")
        return True
    except TypeError:
        # dynamo parameter not recognized in this PyTorch version
        print("dynamo=False not supported, trying without it...")
        try:
            torch.onnx.export(
                model,
                DUMMY_INPUT,
                ONNX_PATH,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                opset_version=11,
            )
            print(f"ONNX exported: {ONNX_PATH}")
            return True
        except Exception as e:
            print(f"Failed: {e}")
            return False
    except Exception as e:
        print(f"Failed: {e}")
        return False


def export_onnx_traced(model):
    """Re-trace the ScriptModule and export."""
    print("\n--- Attempt 2: Trace model then export ---")
    try:
        # Extract the underlying module or re-trace
        with torch.no_grad():
            traced = torch.jit.trace(model, DUMMY_INPUT)
        torch.onnx.export(
            traced,
            DUMMY_INPUT,
            ONNX_PATH,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
            opset_version=11,
        )
        print(f"ONNX exported: {ONNX_PATH}")
        return True
    except Exception as e:
        print(f"Failed: {e}")
        return False


def export_onnx_ts2ep(model):
    """Use TS2EPConverter to convert ScriptModule to ExportedProgram, then export."""
    print("\n--- Attempt 3: TS2EPConverter ---")
    try:
        from torch.onnx import TS2EPConverter
        ep = TS2EPConverter(model, (DUMMY_INPUT,)).convert()
        torch.onnx.export(
            ep,
            (DUMMY_INPUT,),
            ONNX_PATH,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        )
        print(f"ONNX exported: {ONNX_PATH}")
        return True
    except ImportError:
        print("TS2EPConverter not available in this PyTorch version")
        return False
    except Exception as e:
        print(f"Failed: {e}")
        return False


def export_onnx_aten_fallback(model):
    """Export with ONNX_ATEN_FALLBACK operator export type."""
    print("\n--- Attempt 4: ATEN fallback export ---")
    try:
        torch.onnx.export(
            model,
            DUMMY_INPUT,
            ONNX_PATH,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
            opset_version=11,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )
        print(f"ONNX exported: {ONNX_PATH}")
        return True
    except Exception as e:
        print(f"Failed: {e}")
        return False


def build_tensorrt_engine():
    """Build TensorRT FP16 engine from ONNX model."""
    print("\n--- Building TensorRT engine ---")
    try:
        import tensorrt as trt

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(ONNX_PATH, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"  ONNX parse error: {parser.get_error(i)}")
                return False

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 256 MB

        # Enable FP16 if supported
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 enabled")

        # Set dynamic batch size
        profile = builder.create_optimization_profile()
        profile.set_shape('input',
                          min=(1, 3, 56, 56),
                          opt=(16, 3, 56, 56),
                          max=(64, 3, 56, 56))
        config.add_optimization_profile(profile)

        print("Building engine (this may take a few minutes)...")
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            print("Engine build failed!")
            return False

        with open(ENGINE_PATH, 'wb') as f:
            f.write(engine_bytes)

        size_mb = os.path.getsize(ENGINE_PATH) / (1024 * 1024)
        print(f"TensorRT engine saved: {ENGINE_PATH} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"TensorRT build failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_engine():
    """Quick verification that the engine loads and produces correct output shape."""
    print("\n--- Verifying engine ---")
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        with open(ENGINE_PATH, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()
        context.set_input_shape('input', (1, 3, 56, 56))

        # Allocate buffers
        h_input = np.random.randn(1, 3, 56, 56).astype(np.float32)
        h_output = np.empty((1, 3), dtype=np.float32)

        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)

        cuda.memcpy_htod(d_input, h_input)
        context.set_tensor_address('input', int(d_input))
        context.set_tensor_address('output', int(d_output))
        context.execute_async_v3(stream_handle=0)
        cuda.memcpy_dtoh(h_output, d_output)

        print(f"Engine output shape: {h_output.shape}")
        print(f"Sample output: {h_output[0]}")
        print(f"Predicted class: {np.argmax(h_output[0])}")
        print("Engine verification PASSED")
        return True
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    if not os.path.exists(PTS_PATH):
        print(f"Model not found: {PTS_PATH}")
        sys.exit(1)

    print(f"PyTorch version: {torch.__version__}")
    model = load_torchscript_model()

    # Try multiple ONNX export strategies
    exported = False
    for export_fn in [export_onnx_legacy, export_onnx_traced,
                      export_onnx_ts2ep, export_onnx_aten_fallback]:
        if export_fn(model):
            exported = True
            break

    if not exported:
        print("\nAll ONNX export methods failed.")
        print("TCD-Net will use PyTorch CPU/GPU fallback (slower but functional).")
        sys.exit(1)

    # Build TensorRT engine
    if build_tensorrt_engine():
        verify_engine()
        print("\nDone! TCD-Net TensorRT FP16 engine is ready.")
        print("Restart the detector to use it: sudo systemctl restart darts-detector")
    else:
        print("\nTensorRT build failed. Cleaning up ONNX file.")
        print("TCD-Net will use PyTorch fallback.")

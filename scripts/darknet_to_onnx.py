"""Darknet YOLOv4 to ONNX converter.

Parses .cfg and .weights, builds ONNX graph with proper channel tracking.
"""
import os
import sys
import struct
import numpy as np


def parse_cfg(cfg_path):
    """Parse darknet .cfg into list of block dicts."""
    blocks = []
    with open(cfg_path, 'r') as f:
        block = {}
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('['):
                if block:
                    blocks.append(block)
                block = {'type': line[1:-1].strip()}
            else:
                if '=' in line:
                    key, val = line.split('=', 1)
                    block[key.strip()] = val.strip()
        if block:
            blocks.append(block)
    return blocks


def compute_all_channels(blocks):
    """Pre-compute output channels for every layer.

    Returns list where channels[i] is the output channel count of layer i.
    Layer indices are 0-based (block index - 1, since block 0 is [net]).
    """
    net = blocks[0]
    in_channels = int(net.get('channels', 3))
    channels = []  # channels[i] = output channels of layer i

    for i, block in enumerate(blocks[1:]):
        bt = block['type']
        if bt == 'convolutional':
            ch = int(block['filters'])
        elif bt == 'shortcut':
            # Output has same channels as the current input (previous layer)
            ch = channels[-1] if channels else in_channels
        elif bt == 'route':
            layers_str = block['layers']
            route_layers = [int(x.strip()) for x in layers_str.split(',')]
            ch = 0
            for rl in route_layers:
                if rl < 0:
                    idx = i + rl  # relative to current layer (0-based)
                else:
                    idx = rl
                if idx < 0:
                    ch += in_channels
                else:
                    ch += channels[idx]
        elif bt == 'upsample':
            ch = channels[-1] if channels else in_channels
        elif bt == 'maxpool':
            ch = channels[-1] if channels else in_channels
        elif bt == 'yolo':
            ch = channels[-1] if channels else in_channels
        else:
            ch = channels[-1] if channels else in_channels
        channels.append(ch)

    return channels


def load_weights(weights_path):
    """Load darknet weights file."""
    with open(weights_path, 'rb') as f:
        major, minor, revision = struct.unpack('3i', f.read(12))
        if (major * 10 + minor) >= 2:
            seen = struct.unpack('q', f.read(8))[0]
        else:
            seen = struct.unpack('i', f.read(4))[0]
        weights = np.fromfile(f, dtype=np.float32)
    print(f"Weights header: v{major}.{minor}.{revision}, seen={seen}")
    print(f"Total weight values: {len(weights)} ({len(weights)*4/1e6:.1f} MB)")
    return weights


def darknet_to_onnx(cfg_path, weights_path, onnx_path, input_w=416, input_h=416):
    """Convert Darknet YOLOv4 to ONNX."""
    try:
        import onnx
        from onnx import helper, TensorProto, numpy_helper
    except ImportError:
        os.system(f"{sys.executable} -m pip install onnx")
        import onnx
        from onnx import helper, TensorProto, numpy_helper

    blocks = parse_cfg(cfg_path)
    weights = load_weights(weights_path)
    all_channels = compute_all_channels(blocks)

    print(f"\nModel: {len(blocks)-1} layers, input {input_w}x{input_h}")
    print(f"Channel sizes: {all_channels[:10]}... (first 10)")

    nodes = []
    initializers = []
    inputs = [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, input_h, input_w])]
    outputs_info = []

    layer_outputs = {}  # layer_index -> output_name
    weight_offset = 0
    output_name = 'input'
    yolo_count = 0
    in_channels = int(blocks[0].get('channels', 3))

    for layer_idx, block in enumerate(blocks[1:]):
        block_type = block['type']
        prev_output = output_name

        # Determine input channels for this layer
        if layer_idx == 0:
            inp_ch = in_channels
        else:
            # For route layers, inp_ch is computed differently
            if block_type == 'route':
                inp_ch = None  # route doesn't have a single input
            else:
                inp_ch = all_channels[layer_idx - 1]

        if block_type == 'convolutional':
            filters = int(block['filters'])
            size = int(block['size'])
            stride = int(block['stride'])
            pad = int(block.get('pad', 0))
            padding = size // 2 if pad else 0
            batch_normalize = int(block.get('batch_normalize', 0))
            activation = block.get('activation', 'linear')

            conv_name = f'conv_{layer_idx}'

            if batch_normalize:
                # BN: bias, scale, running_mean, running_var
                bn_bias = weights[weight_offset:weight_offset + filters].copy()
                weight_offset += filters
                bn_scale = weights[weight_offset:weight_offset + filters].copy()
                weight_offset += filters
                bn_mean = weights[weight_offset:weight_offset + filters].copy()
                weight_offset += filters
                bn_var = weights[weight_offset:weight_offset + filters].copy()
                weight_offset += filters

                n_weights = filters * inp_ch * size * size
                conv_w = weights[weight_offset:weight_offset + n_weights].copy()
                weight_offset += n_weights
                conv_w = conv_w.reshape(filters, inp_ch, size, size)

                w_name = f'{conv_name}_w'
                initializers.append(numpy_helper.from_array(conv_w, w_name))
                conv_out = f'{conv_name}_out'
                nodes.append(helper.make_node('Conv', [prev_output, w_name], [conv_out],
                    kernel_shape=[size, size], strides=[stride, stride],
                    pads=[padding, padding, padding, padding], name=conv_name))

                bn_name = f'bn_{layer_idx}'
                for arr, suffix in [(bn_scale, '_scale'), (bn_bias, '_bias'),
                                     (bn_mean, '_mean'), (bn_var, '_var')]:
                    initializers.append(numpy_helper.from_array(arr, f'{bn_name}{suffix}'))

                bn_out = f'{bn_name}_out'
                nodes.append(helper.make_node('BatchNormalization',
                    [conv_out, f'{bn_name}_scale', f'{bn_name}_bias',
                     f'{bn_name}_mean', f'{bn_name}_var'],
                    [bn_out], epsilon=1e-5, name=bn_name))
                output_name = bn_out
            else:
                # Conv with bias (no BN) — typically the last conv before YOLO
                conv_bias = weights[weight_offset:weight_offset + filters].copy()
                weight_offset += filters
                n_weights = filters * inp_ch * size * size
                conv_w = weights[weight_offset:weight_offset + n_weights].copy()
                weight_offset += n_weights
                conv_w = conv_w.reshape(filters, inp_ch, size, size)

                w_name = f'{conv_name}_w'
                b_name = f'{conv_name}_b'
                initializers.append(numpy_helper.from_array(conv_w, w_name))
                initializers.append(numpy_helper.from_array(conv_bias, b_name))

                conv_out = f'{conv_name}_out'
                nodes.append(helper.make_node('Conv', [prev_output, w_name, b_name], [conv_out],
                    kernel_shape=[size, size], strides=[stride, stride],
                    pads=[padding, padding, padding, padding], name=conv_name))
                output_name = conv_out

            # Activation
            if activation == 'leaky':
                act_out = f'leaky_{layer_idx}_out'
                nodes.append(helper.make_node('LeakyRelu', [output_name], [act_out],
                    alpha=0.1, name=f'leaky_{layer_idx}'))
                output_name = act_out
            elif activation == 'mish':
                sp_out = f'sp_{layer_idx}_out'
                nodes.append(helper.make_node('Softplus', [output_name], [sp_out],
                    name=f'softplus_{layer_idx}'))
                tanh_out = f'tanh_{layer_idx}_out'
                nodes.append(helper.make_node('Tanh', [sp_out], [tanh_out],
                    name=f'tanh_{layer_idx}'))
                mish_out = f'mish_{layer_idx}_out'
                nodes.append(helper.make_node('Mul', [output_name, tanh_out], [mish_out],
                    name=f'mish_{layer_idx}'))
                output_name = mish_out
            elif activation == 'logistic':
                sig_out = f'sig_{layer_idx}_out'
                nodes.append(helper.make_node('Sigmoid', [output_name], [sig_out],
                    name=f'sigmoid_{layer_idx}'))
                output_name = sig_out

        elif block_type == 'shortcut':
            from_layer = int(block['from'])
            if from_layer < 0:
                from_idx = layer_idx + from_layer
            else:
                from_idx = from_layer
            add_out = f'shortcut_{layer_idx}_out'
            nodes.append(helper.make_node('Add',
                [prev_output, layer_outputs[from_idx]], [add_out],
                name=f'shortcut_{layer_idx}'))
            output_name = add_out
            act = block.get('activation', 'linear')
            if act == 'leaky':
                act_out = f'shortcut_leaky_{layer_idx}_out'
                nodes.append(helper.make_node('LeakyRelu', [output_name], [act_out],
                    alpha=0.1, name=f'shortcut_leaky_{layer_idx}'))
                output_name = act_out

        elif block_type == 'route':
            layers_str = block['layers']
            route_layers = [int(x.strip()) for x in layers_str.split(',')]
            resolved = []
            for rl in route_layers:
                if rl < 0:
                    resolved.append(layer_idx + rl)
                else:
                    resolved.append(rl)
            if len(resolved) == 1:
                output_name = layer_outputs[resolved[0]]
            else:
                concat_out = f'route_{layer_idx}_out'
                nodes.append(helper.make_node('Concat',
                    [layer_outputs[r] for r in resolved], [concat_out],
                    axis=1, name=f'route_{layer_idx}'))
                output_name = concat_out

        elif block_type == 'upsample':
            stride_val = int(block['stride'])
            up_out = f'upsample_{layer_idx}_out'
            roi_name = f'up_{layer_idx}_roi'
            scales_name = f'up_{layer_idx}_scales'
            initializers.append(numpy_helper.from_array(
                np.array([], dtype=np.float32), roi_name))
            initializers.append(numpy_helper.from_array(
                np.array([1.0, 1.0, float(stride_val), float(stride_val)], dtype=np.float32),
                scales_name))
            nodes.append(helper.make_node('Resize',
                [output_name, roi_name, scales_name], [up_out],
                mode='nearest', name=f'upsample_{layer_idx}'))
            output_name = up_out

        elif block_type == 'maxpool':
            mp_size = int(block['size'])
            mp_stride = int(block['stride'])
            mp_pad = (mp_size - 1) // 2
            mp_out = f'maxpool_{layer_idx}_out'
            nodes.append(helper.make_node('MaxPool', [output_name], [mp_out],
                kernel_shape=[mp_size, mp_size],
                strides=[mp_stride, mp_stride],
                pads=[mp_pad, mp_pad, mp_pad, mp_pad],
                name=f'maxpool_{layer_idx}'))
            output_name = mp_out

        elif block_type == 'yolo':
            yolo_out = f'yolo_{yolo_count}'
            nodes.append(helper.make_node('Identity', [output_name], [yolo_out],
                name=f'yolo_id_{yolo_count}'))
            outputs_info.append(helper.make_tensor_value_info(yolo_out, TensorProto.FLOAT, None))
            yolo_count += 1
            output_name = yolo_out

        layer_outputs[layer_idx] = output_name

        if block_type == 'convolutional' and layer_idx % 20 == 0:
            print(f"  Layer {layer_idx}: {block_type} inp_ch={inp_ch} -> {filters}, "
                  f"weights consumed: {weight_offset}")

    print(f"\nParsed {layer_idx + 1} layers")
    print(f"Weights consumed: {weight_offset}/{len(weights)} "
          f"({weight_offset*100/len(weights):.1f}%)")

    if abs(weight_offset - len(weights)) > 10:
        print(f"WARNING: {len(weights) - weight_offset} unused weights!")
        print("This may indicate incorrect channel tracking.")

    graph = helper.make_graph(nodes, 'yolov4', inputs, outputs_info, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    model.ir_version = 7

    onnx.save(model, onnx_path)
    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"\nONNX saved: {onnx_path} ({size_mb:.1f} MB)")
    print(f"  Inputs: {[i.name for i in model.graph.input]}")
    print(f"  Outputs: {[o.name for o in model.graph.output]}")


if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_dir, 'models')

    cfg = os.path.join(models_dir, 'yolov4-custom.cfg')
    wts = os.path.join(models_dir, 'yolov4-custom_final.weights')
    out = os.path.join(models_dir, 'yolov4-custom.onnx')

    darknet_to_onnx(cfg, wts, out, 416, 416)

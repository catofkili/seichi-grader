#!/usr/bin/env python3
"""Weight-only int8 quantization for the ISNet anime matting models.

fp16 权重砍半的思路：全图先回到 fp32（wasm CPU 本就无 fp16 卷积核心，
现网 70 个 Cast 实际是在 fp32 里绕圈），再把 114 个 Conv 的权重按输出通道
对称量化成 int8 + DequantizeLinear。激活保持浮点，精度损失应远小于
静态全 int8；体积约减半（fp16 84MB → int8 权重 ~44MB）。

用法:
  .venv-convert/bin/python tools/quantize-isnet-weights.py models/isnet-anime-fp16.onnx
产出: models/<name>-w8.onnx（如 isnet-anime-w8.onnx）
"""
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, version_converter

ROOT = Path(__file__).resolve().parent.parent


def to_fp32(model: onnx.ModelProto) -> onnx.ModelProto:
    """把 fp16 混合图整体转成 fp32：权重升精度，Cast→fp16 改为 Cast→fp32。"""
    g = model.graph
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype(np.float32)
            init.CopyFrom(numpy_helper.from_array(arr, init.name))
    for node in g.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.FLOAT
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        tt = vi.type.tensor_type
        if tt.elem_type == TensorProto.FLOAT16:
            tt.elem_type = TensorProto.FLOAT
    return model


def quantize_conv_weights(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Conv 权重 per-channel 对称 int8 + DequantizeLinear，返回(模型, 量化层数)。"""
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    new_nodes = []
    n_quant = 0
    for node in g.node:
        if node.op_type == "Conv" and node.input[1] in inits:
            w_init = inits[node.input[1]]
            w = numpy_helper.to_array(w_init)  # [O, I, kH, kW] fp32
            absmax = np.abs(w).max(axis=(1, 2, 3))
            scale = np.where(absmax > 0, absmax / 127.0, 1.0).astype(np.float32)
            q = np.clip(np.round(w / scale[:, None, None, None]), -127, 127).astype(np.int8)

            qname = w_init.name + "_q8"
            g.initializer.append(numpy_helper.from_array(q, qname))
            g.initializer.append(numpy_helper.from_array(scale, w_init.name + "_scale"))
            g.initializer.append(
                numpy_helper.from_array(np.zeros(len(scale), dtype=np.int8), w_init.name + "_zp"))
            dq_out = w_init.name + "_dq"
            new_nodes.append(helper.make_node(
                "DequantizeLinear",
                [qname, w_init.name + "_scale", w_init.name + "_zp"], [dq_out],
                name=w_init.name + "_dql", axis=0))
            node.input[1] = dq_out
            g.initializer.remove(w_init)
            n_quant += 1
    # DequantizeLinear 节点放最前（初值就绪即可算，ORT 会按拓扑排序执行）
    g.node.extend([])  # noop, keep type
    all_nodes = new_nodes + list(g.node)
    del g.node[:]
    g.node.extend(all_nodes)
    return model, n_quant


def main() -> None:
    src = Path(sys.argv[1])
    model = onnx.load(str(src))
    model = to_fp32(model)
    model = version_converter.convert_version(model, 13)  # per-channel DQL 需 opset 13
    model, n = quantize_conv_weights(model)
    onnx.checker.check_model(model)
    out = src.with_name(src.stem.replace("-fp16", "") + "-w8.onnx")
    onnx.save(model, str(out))
    print(f"量化 Conv 层数: {n}")
    print(f"输入: {src.name} {src.stat().st_size/1048576:.1f}MB")
    print(f"输出: {out.name} {out.stat().st_size/1048576:.1f}MB")


if __name__ == "__main__":
    main()

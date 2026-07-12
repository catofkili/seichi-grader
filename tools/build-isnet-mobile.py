#!/usr/bin/env python3
"""Build the 512px mobile ISNet ONNX from the project's 1024px FP16 model."""
from pathlib import Path
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models/isnet-anime-fp16.onnx"
OUTPUT = ROOT / "models/isnet-anime-512-fp16.onnx"

model = onnx.load(SOURCE)
input_shape = model.graph.input[0].type.tensor_type.shape.dim
input_shape[2].dim_value = 512
input_shape[3].dim_value = 512
initializers = {value.name: value for value in model.graph.initializer}
changed = 0

for node in model.graph.node:
    if node.op_type != "Resize" or len(node.input) < 4 or node.input[3] not in initializers:
        continue
    tensor = initializers[node.input[3]]
    sizes = numpy_helper.to_array(tensor).copy()
    if sizes.size != 4:
        continue
    sizes[-2:] //= 2
    tensor.CopyFrom(numpy_helper.from_array(sizes, tensor.name))
    changed += 1

if changed != 34:
    raise RuntimeError(f"Expected 34 fixed Resize nodes, found {changed}; source model changed")

# Stale 1024px annotations conflict with runtime inference after resizing.
for value in [*model.graph.value_info, *model.graph.output]:
    for dim in value.type.tensor_type.shape.dim:
        dim.ClearField("dim_value")
        dim.ClearField("dim_param")

onnx.checker.check_model(model)
onnx.save(model, OUTPUT)
print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1048576:.1f} MiB)")

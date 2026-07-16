import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(4,8),
    nn.ReLU(),
    nn.Linear(8,2)
)

dummy_input = torch.randn(1,4)

torch.onnx.export(
    model,
    dummy_input,
    "simple_model.onnx",
    input_names=["input"],
    output_names=["output"]
)

print("ONNX model exported successfully.")
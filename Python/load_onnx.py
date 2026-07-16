import onnx

model = onnx.load("simple_model.onnx")

onnx.checker.check_model(model)

print("ONNX model is valid.")
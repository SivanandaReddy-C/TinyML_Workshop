import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("simple_model.onnx")

input_name = session.get_inputs()[0].name

x = np.random.randn(1,4).astype(np.float32)

output = session.run(None,{input_name:x})

print(output)
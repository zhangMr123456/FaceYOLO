import onnxruntime

print(onnxruntime.get_device())
print(onnxruntime.get_available_providers())

import torch

print(torch.cuda.is_available())
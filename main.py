import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
props = torch.cuda.get_device_properties(device)
print(f"GPU Name: {props.name}")
print(f"Compute Capability: {props.major}.{props.minor}")

# Kiểm tra hỗ trợ bfloat16
print("Supports BF16:", torch.cuda.is_bf16_supported())

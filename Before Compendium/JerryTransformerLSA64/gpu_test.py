import torch
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
print("torch.backends.cudnn.version():", torch.backends.cudnn.version() if torch.cuda.is_available() else None)

import torch

from child_cnn_linear import compute_padding

# test padding computation
height = 31
width = 31
kernel_height = 2
kernel_width = 2
stride_height = 2
stride_width = 3

top, bottom, left, right = compute_padding(height, width, kernel_height, kernel_width, stride_height, stride_width)
print(f"Top: {top}")
print(f"bottom: {bottom}")
print(f"Left: {left}")
print(f"right: {right}")

image = torch.randn(1, 1, height, width)
padding_layer = torch.nn.ZeroPad2d((left, right, top, bottom))

conv_layer = torch.nn.Conv2d(
            1, 1,
            kernel_size=(kernel_height, kernel_width),
            stride=(stride_height, stride_width))
output = conv_layer(padding_layer(image))
print(f"output shape: {output.size()}")

import torch

from child_cnn_linear import compute_padding

# test padding computation
# height = 31
# width = 31
# kernel_height = 2
# kernel_width = 2
# stride_height = 2
# stride_width = 3

# top, bottom, left, right = compute_padding(height, width, kernel_height, kernel_width, stride_height, stride_width)
# print(f"Top: {top}")
# print(f"bottom: {bottom}")
# print(f"Left: {left}")
# print(f"right: {right}")

# image = torch.randn(1, 1, height, width)
# padding_layer = torch.nn.ZeroPad2d((left, right, top, bottom))

# conv_layer = torch.nn.Conv2d(
#             1, 1,
#             kernel_size=(kernel_height, kernel_width),
#             stride=(stride_height, stride_width))
# output = conv_layer(padding_layer(image))
# print(f"output shape: {output.size()}")

# test layers not wrapped in model
class CNN(nn.Module):
    def __init__(self, conv_layers, pool_layers, drop_layers, fc_layers):
        super(CNN, self).__init__()
        self.conv_layers = conv_layers
        self.pool_layers = pool_layers
        self.drop_layers = drop_layers
        self.fc_layers = fc_layers
        for i in range(len(conv_layers)):
            setattr(self, 'conv{}'.format(i+1), pool_layers[i])
            # setattr(self, f'conv{i+1}', conv_layers[i])
        for i in range(len(pool_layers)):
            if pool_layers[i]:
                setattr(self, 'pool{}'.format(i+1), pool_layers[i])
        for i in range(len(drop_layers)):
            if drop_layers[i]:
                setattr(self, 'drop{}'.format(i+1), drop_layers[i])
        for j in range(len(fc_layers)):
            setattr(self, 'fc{}'.format(j+1), fc_layers[j])

    def forward(self, x):
        x = conv_forward(x, self.conv_layers, self.pool_layers, self.drop_layers)
        x = x.view(x.size()[0], -1)
        x = fc_forward(x, self.fc_layers)
        return x

'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

import torch.utils.data
from torch.nn import functional as F

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple





def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
	input_rows = input.size(2)
	filter_rows = weight.size(2)
	effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
	out_rows = (input_rows + stride[0] - 1) // stride[0]
	padding_rows = max(0, (out_rows - 1) * stride[0] +
						(filter_rows - 1) * dilation[0] + 1 - input_rows)
	rows_odd = (padding_rows % 2 != 0)
	padding_cols = max(0, (out_rows - 1) * stride[0] +
						(filter_rows - 1) * dilation[0] + 1 - input_rows)
	cols_odd = (padding_rows % 2 != 0)

	if rows_odd or cols_odd:
		input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

	return F.conv2d(input, weight, bias, stride,
				  padding=(padding_rows // 2, padding_cols // 2),
				  dilation=dilation, groups=groups)

class _ConvNd(Module):

	def __init__(self, in_channels, out_channels, kernel_size, stride,
				 padding, dilation, transposed, output_padding, groups, bias):
		super(_ConvNd, self).__init__()
		if in_channels % groups != 0:
			raise ValueError('in_channels must be divisible by groups')
		if out_channels % groups != 0:
			raise ValueError('out_channels must be divisible by groups')
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.dilation = dilation
		self.transposed = transposed
		self.output_padding = output_padding
		self.groups = groups
		if transposed:
			self.weight = Parameter(torch.Tensor(
				in_channels, out_channels // groups, *kernel_size))
		else:
			self.weight = Parameter(torch.Tensor(
				out_channels, in_channels // groups, *kernel_size))
		if bias:
			self.bias = Parameter(torch.Tensor(out_channels))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		n = self.in_channels
		for k in self.kernel_size:
			n *= k
		stdv = 1. / math.sqrt(n)
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def __repr__(self):
		s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
			 ', stride={stride}')
		if self.padding != (0,) * len(self.padding):
			s += ', padding={padding}'
		if self.dilation != (1,) * len(self.dilation):
			s += ', dilation={dilation}'
		if self.output_padding != (0,) * len(self.output_padding):
			s += ', output_padding={output_padding}'
		if self.groups != 1:
			s += ', groups={groups}'
		if self.bias is None:
			s += ', bias=False'
		s += ')'
		return s.format(name=self.__class__.__name__, **self.__dict__)

class Conv2d(_ConvNd):

	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
				 padding=0, dilation=1, groups=1, bias=True):
		kernel_size = _pair(kernel_size)
		stride = _pair(stride)
		padding = _pair(padding)
		dilation = _pair(dilation)
		super(Conv2d, self).__init__(
			in_channels, out_channels, kernel_size, stride, padding, dilation,
			False, _pair(0), groups, bias)


	def forward(self, input):
		return conv2d_same_padding(input, self.weight, self.bias, self.stride,
						self.padding, self.dilation, self.groups)


class NAS15(nn.Module):
	def __init__(self):
		super(NAS15, self).__init__()
		self.cov1 = Conv2d(3,36,(3,3))
		self.batch1 = nn.BatchNorm2d(36)
		self.cov2 = Conv2d(36,48,(3,3))
		self.batch2 = nn.BatchNorm2d(48)
		self.cov3 = Conv2d(84,36,(3,3))
		self.batch3 = nn.BatchNorm2d(36)
		self.cov4 = Conv2d(120,36,(5,5))
		self.batch4 = nn.BatchNorm2d(36)
		self.cov5 = nn.Conv2d(72,48,(3,7),1,(1,3))
		self.batch5 = nn.BatchNorm2d(48)
		self.cov6 = Conv2d(168,48,(7,7))
		self.batch6 = nn.BatchNorm2d(48)
		self.cov7 = Conv2d(216,48,(7,7))
		self.batch7 = nn.BatchNorm2d(48)
		self.cov8 = nn.Conv2d(132,36,(7,3),1,(3,1))
		self.batch8 = nn.BatchNorm2d(36)
		self.cov9 = nn.Conv2d(168,36,(7,1),1,(3,0))
		self.batch9 = nn.BatchNorm2d(36)
		self.cov10 = Conv2d(324,36,(7,7))
		self.batch10 = nn.BatchNorm2d(36)
		self.cov11 = nn.Conv2d(336,36,(5,7),1,(2,3))
		self.batch11 = nn.BatchNorm2d(36)
		self.cov12 = Conv2d(240,48,(7,7))
		self.batch12 = nn.BatchNorm2d(48)
		self.cov13 = nn.Conv2d(360,48,(7,5),1,(3,2))
		self.batch13 = nn.BatchNorm2d(48)
		self.cov14 = nn.Conv2d(180,48,(7,5),1,(3,2))
		self.batch14 = nn.BatchNorm2d(48)
		self.cov15 = nn.Conv2d(276,48,(7,5),1,(3,2))
		self.batch15 = nn.BatchNorm2d(48)

		self.classifier = nn.Linear(49152, 10)



	def forward(self, x, place_holder=None):
                o1 =  (F.relu(self.cov1(x)))
                o2 =  (F.relu(self.cov2(o1)))
                co12 = torch.cat((o1, o2), 1)
                o3 =  (F.relu(self.cov3(co12)))
                co123 = torch.cat((o1, o2, o3), 1)
                o4 =  (F.relu(self.cov4(co123)))
                co34 = torch.cat((o3, o4), 1)
                o5 =  (F.relu(self.cov5(co34)))
                co2345 = torch.cat((o2, o3, o4, o5), 1)
                o6 =  (F.relu(self.cov6(co2345)))
                co23456 = torch.cat((o2, o3, o4, o5, o6), 1)
                o7 =  (F.relu(self.cov7(co23456)))
                co167 = torch.cat((o1, o6, o7), 1)
                o8 =  (F.relu(self.cov8(co167)))
                co1658 = torch.cat((o1, o6, o5, o8), 1)
                o9 =  (F.relu(self.cov9(co1658)))
                co13456789 = torch.cat((o1, o3, o4, o5, o6, o7, o8, o9), 1)
                o10 =  (F.relu(self.cov10(co13456789)))
                co125678910 = torch.cat((o1, o2, o5, o6, o7, o8, o9, o10), 1)
                o11 =  (F.relu(self.cov11(co125678910)))
                co1234611 = torch.cat((o1, o2, o3, o4, o6, o11), 1)
                o12 =  (F.relu(self.cov12(co1234611)))
                co136789101112 = torch.cat((o1, o3, o6, o7, o8, o9, o10, o11, o12), 1)
                o13 =  (F.relu(self.cov13(co136789101112)))
                co371213 = torch.cat((o3, o7, o12, o13), 1)
                o14 =  (F.relu(self.cov14(co371213)))
                co2711121314 = torch.cat((o2, o7, o11, o12, o13, o14), 1)
                o15 =  (F.relu(self.cov15(co2711121314)))

                o15 = o15.view(o15.size(0), -1)
                out = self.classifier(o15)

                return out




def test():
	net = NAS15()
	x = torch.randn(1,3,32,32)
	y = net(x)
	print(y.size())


#test()


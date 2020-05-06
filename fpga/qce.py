import numpy as np
from config import ARCH_SPACE, QUAN_SPACE
import math

MULT_LUT = np.zeros((15, 15))
MULT_LUT[1][1] = 1;
MULT_LUT[1][2] = 3;
MULT_LUT[2][2] = 4;
MULT_LUT[1][3] = 5;
MULT_LUT[2][3] = 11;
MULT_LUT[3][3] = 17;
MULT_LUT[1][4] = 6;
MULT_LUT[2][4] = 14;
MULT_LUT[3][4] = 21;
MULT_LUT[4][4] = 29;
MULT_LUT[1][5] = 7;
MULT_LUT[2][5] = 17;
MULT_LUT[3][5] = 26;
MULT_LUT[4][5] = 34;
MULT_LUT[5][5] = 41;
MULT_LUT[1][6] = 8;
MULT_LUT[2][6] = 20;
MULT_LUT[3][6] = 32;
MULT_LUT[4][6] = 39;
MULT_LUT[5][6] = 47;
MULT_LUT[6][6] = 58;
MULT_LUT[1][7] = 9;
MULT_LUT[2][7] = 23;
MULT_LUT[3][7] = 36;
MULT_LUT[4][7] = 44;
MULT_LUT[5][7] = 53;
MULT_LUT[6][7] = 65;
MULT_LUT[7][7] = 74;
MULT_LUT[1][8] = 10;
MULT_LUT[2][8] = 26;
MULT_LUT[3][8] = 41;
MULT_LUT[4][8] = 49;
MULT_LUT[5][8] = 59;
MULT_LUT[6][8] = 72;
MULT_LUT[7][8] = 82;
MULT_LUT[8][8] = 95;
MULT_LUT[1][9] = 11;
MULT_LUT[2][9] = 29;
MULT_LUT[3][9] = 46;
MULT_LUT[4][9] = 54;
MULT_LUT[5][9] = 65;
MULT_LUT[6][9] = 79;
MULT_LUT[7][9] = 92;
MULT_LUT[8][9] = 104;
MULT_LUT[9][9] = 118;
MULT_LUT[1][10] = 12;
MULT_LUT[2][10] = 32;
MULT_LUT[3][10] = 51;
MULT_LUT[4][10] = 59;
MULT_LUT[5][10] = 73;
MULT_LUT[6][10] = 86;
MULT_LUT[7][10] = 100;
MULT_LUT[8][10] = 113;
MULT_LUT[9][10] = 128;
MULT_LUT[10][10] = 140;
MULT_LUT[1][11] = 13;
MULT_LUT[2][11] = 35;
MULT_LUT[3][11] = 56;
MULT_LUT[4][11] = 64;
MULT_LUT[5][11] = 79;
MULT_LUT[6][11] = 93;
MULT_LUT[7][11] = 108;
MULT_LUT[8][11] = 122;
MULT_LUT[9][11] = 138;
MULT_LUT[10][11] = 152;
MULT_LUT[11][11] = 167;
MULT_LUT[1][12] = 14;
MULT_LUT[2][12] = 38;
MULT_LUT[3][12] = 61;
MULT_LUT[4][12] = 69;
MULT_LUT[5][12] = 85;
MULT_LUT[6][12] = 100;
MULT_LUT[7][12] = 116;
MULT_LUT[8][12] = 131;
MULT_LUT[9][12] = 148;
MULT_LUT[10][12] = 163;
MULT_LUT[11][12] = 179;
MULT_LUT[12][12] = 195;
MULT_LUT[1][13] = 15;
MULT_LUT[2][13] = 41;
MULT_LUT[3][13] = 66;
MULT_LUT[4][13] = 74;
MULT_LUT[5][13] = 91;
MULT_LUT[6][13] = 107;
MULT_LUT[7][13] = 124;
MULT_LUT[8][13] = 42;
MULT_LUT[9][13] = 158;
MULT_LUT[10][13] = 174;
MULT_LUT[11][13] = 191;
MULT_LUT[12][13] = 208;
MULT_LUT[13][13] = 224;
MULT_LUT[1][14] = 16;
MULT_LUT[2][14] = 44;
MULT_LUT[3][14] = 71;
MULT_LUT[4][14] = 79;
MULT_LUT[5][14] = 97;
MULT_LUT[6][14] = 114;
MULT_LUT[7][14] = 132;
MULT_LUT[8][14] = 149;
MULT_LUT[9][14] = 168;
MULT_LUT[10][14] = 185;
MULT_LUT[11][14] = 204;
MULT_LUT[12][14] = 221;
MULT_LUT[13][14] = 238;
MULT_LUT[14][14] = 255;

# for i in range(1, 15):
# 	for j in range(1, i):
# 		MULT_LUT[i][j] = MULT_LUT[j][i]

# print(MULT_LUT)


ADDER_LUT = [float('Nan')] * 100
for i in range(len(ADDER_LUT)):
	ADDER_LUT[i] = i

num_layers = 8
max_num_channels = max(ARCH_SPACE['num_filters'])
max_num_channels *= (num_layers - 1)
max_num_int_bits = max(QUAN_SPACE['act_num_int_bits'])+1+\
				max(QUAN_SPACE['weight_num_int_bits'])
max_num_frac_bits = max(QUAN_SPACE['act_num_frac_bits'])+\
				max(QUAN_SPACE['weight_num_frac_bits'])

min_num_int_bits = min(QUAN_SPACE['act_num_int_bits']) + min(QUAN_SPACE['weight_num_int_bits'])
min_num_frac_bits = min(QUAN_SPACE['act_num_frac_bits']) + min(QUAN_SPACE['weight_num_frac_bits'])

max_num_int_bits += (math.ceil(math.log(max_num_channels, 2)) + 1)

# max_num_int = num_int_bits * 2 + math.ceil(math.log(max_num_channels, 2)) + 1
max_delta_num_int_bits = max(abs(max_num_int_bits - min(QUAN_SPACE['act_num_int_bits'])),
					abs(max(QUAN_SPACE['act_num_int_bits']) - min_num_int_bits))
max_delta_num_frac_bits = max(abs(max_num_frac_bits - min(QUAN_SPACE['act_num_frac_bits'])),
					abs(max(QUAN_SPACE['act_num_frac_bits']) - min_num_frac_bits))

# print(max_delta_num_frac_bits)
TRUNCATOR_LUT = np.zeros((max_num_int_bits+1, max_num_frac_bits+1, max_delta_num_int_bits+1, max_delta_num_frac_bits+1))
# print(TRUNCATOR_LUT.shape)
TRUNCATOR_LUT[3][2][1][0] = 3
base = TRUNCATOR_LUT[3][2][1][0]
for i in range(3, max_num_int_bits+1):
	basei = base + (i-3)
	for j in range(2, max_num_frac_bits+1):
		basej = basei + (j-2)
		for k in range(0, max_delta_num_int_bits+1):
			basek = basej - (k > 1)
			for l in range(0, max_delta_num_frac_bits+1):
				# print(i, j, k, l)
				TRUNCATOR_LUT[i][j][k][l] = basek - l
				if k == i - 1 and l == j - 1:
					TRUNCATOR_LUT[i][j][k][l] -= 1

# LUT_PER_PROCESSOR = np.zeros((max_num_channels+1, max_num_channels+1, max_num_bits+1, max_num_bits+1))

def compute_ce_size(Tn, Tm, Bii, Bif, Boi, Bof, Bwi, Bwf):
	act_num_int_bits = Bii
	act_num_frac_bits = Bif
	weight_num_int_bits = Bwi
	weight_num_frac_bits = Bwf
	act_num_bits = act_num_int_bits + act_num_frac_bits
	weight_num_bits = weight_num_int_bits + weight_num_frac_bits
	# num_int_bits = max(act_num_int_bits, weight_num_int_bits)
	# num_frac_bits = max(act_num_frac_bits, weight_num_frac_bits)
	# print(Tn, Tm, Bi, Bo)
	num_multipliers = Tn * Tm
	# print(num_int_bits + num_frac_bits)
	multiplier_LUT = MULT_LUT[act_num_int_bits][weight_num_bits] * num_multipliers
	# num_frac_bits = Bi - num_int_bits
	product_num_bits = act_num_bits + weight_num_bits
	num_int_bits = act_num_int_bits + weight_num_int_bits
	num_frac_bits = act_num_frac_bits + weight_num_frac_bits
	num_products = Tn
	adder_LUT = 0
	while num_products > 1:
		num_adders = math.floor(num_products/2)
		adder_size = ADDER_LUT[num_frac_bits + num_int_bits]
		adder_LUT += adder_size * num_adders * Tm
		num_products = math.ceil(num_products/2)
		num_int_bits += 1
		# print("number of products: ", num_products)
		# print("number of int bits: ", num_int_bits)
	num_output_int_bits = Boi
	num_output_frac_bits = Bof
	num_int_bits = max(num_int_bits, num_output_int_bits)
	num_frac_bits = max(num_frac_bits, num_output_frac_bits)
	# print("number of int bits: ", num_int_bits)
	# print("number of frac bits: ", num_frac_bits)
	adder_size = ADDER_LUT[num_int_bits + num_frac_bits]
	adder_LUT += adder_size * Tm
	num_int_bits += 1
	int_ = max(num_int_bits, num_output_int_bits)
	delta_int_ = abs(num_int_bits - num_output_int_bits)
	frac_ = max(num_frac_bits, num_output_frac_bits)
	delta_frac_ = abs(num_frac_bits - num_output_frac_bits)

	# print(frac_)
	truncator_size = TRUNCATOR_LUT[int_, frac_, delta_int_, delta_frac_]
	truncator_LUT = truncator_size * Tm
	return multiplier_LUT + adder_LUT + truncator_LUT

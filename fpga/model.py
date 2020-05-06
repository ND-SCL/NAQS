import copy as cp
import math

from config import CLOCK_FREQUENCY
from .qce import compute_ce_size


def _get_hw_paras(arch_paras, quan_paras, input_shape=(3, 32, 32)):
    input_channels, input_height, input_width = input_shape
    output_height, output_width, output_channels = [], [], []
    if_used = [False] * len(arch_paras)
    hw_paras = []
    for l in range(len(arch_paras)):
        pool_size = \
            arch_paras[l]['pool_size'] if 'pool_size' in arch_paras[l] else 1
        para_dict = {}
        stride_height = arch_paras[l]['stride_height'] \
            if 'stride_height' in arch_paras[l] else 1
        stride_width = arch_paras[l]['stride_width'] \
            if 'stride_width' in arch_paras[l] else 1
        if 'anchor_point' in arch_paras[l]:
            para_dict['N'] = 0
            para_dict['R'], para_dict['C'] = 0, 0
            para_dict['Bii'], para_dict['Bif'] = 0, 0
            for i in range(len(arch_paras[l]['anchor_point'])):
                para_dict['N'] += \
                    arch_paras[l]['anchor_point'][i] * output_channels[i]
                para_dict['R'] = max(para_dict['R'], output_height[i])
                para_dict['C'] = max(para_dict['C'], output_width[i])
                para_dict['Bii'] = max(para_dict['Bii'], hw_paras[i]['Boi'])
                para_dict['Bif'] = max(para_dict['Bif'], hw_paras[i]['Bof'])
                if arch_paras[l]['anchor_point'][i] == 1:
                    if_used[i] = True
            if (l+1) == len(arch_paras):
                for i in range(l):
                    if if_used[i] is False:
                        para_dict['N'] += output_channels[i]
                        para_dict['R'] = max(para_dict['R'], output_height[i])
                        para_dict['C'] = max(para_dict['C'], output_width[i])
                        para_dict['Bii'] = max(para_dict['Bii'],
                                               hw_paras[i]['Boi'])
                        para_dict['Bif'] = max(para_dict['Bif'],
                                               hw_paras[i]['Bof'])
            if para_dict['N'] == 0:
                para_dict['N'] = input_channels
                para_dict['R'], para_dict['C'] = input_height, input_width
                para_dict['Bii'],  para_dict['Bif'] = 1, 6
        else:
            para_dict['N'] = \
                output_channels[l-1] if l > 0 else input_channels
            para_dict['R'] = output_height[l-1] if l > 0 else input_height
            para_dict['C'] = output_width[l-1] if l > 0 else input_width
            para_dict['Bii'] = hw_paras[l-1]['Boi'] if l > 0 else 1
            para_dict['Bif'] = hw_paras[l-1]['Bof'] if l > 0 else 6
        padding_height, out_height = compute_padding(
                        para_dict['R'], arch_paras[l]['filter_height'],
                        stride_height
                        )
        padding_width, out_width = compute_padding(
                    para_dict['C'], arch_paras[l]['filter_width'],
                    stride_width
                    )
        para_dict['R'] += padding_height
        para_dict['C'] += padding_width
        _, out_height = compute_padding(out_height, pool_size, pool_size)
        _, out_width = compute_padding(out_width, pool_size, pool_size)
        para_dict['M'] = arch_paras[l]['num_filters']
        output_channels.append(arch_paras[l]['num_filters'])
        output_height.append(out_height)
        output_width.append(out_width)
        para_dict['Kh'] = arch_paras[l]['filter_height']
        para_dict['Kw'] = arch_paras[l]['filter_width']
        para_dict['Boi'] = quan_paras[l]['act_num_int_bits']
        para_dict['Bof'] = quan_paras[l]['act_num_frac_bits']
        para_dict['Bwi'] = quan_paras[l]['weight_num_int_bits']
        para_dict['Bwf'] = quan_paras[l]['weight_num_frac_bits']
        hw_paras.append(para_dict)
    return hw_paras


def compute_padding(input_size, kernel_size, stride):
    output_size = math.floor((input_size + stride - 1) / stride)
    padding = max(0, (output_size-1) * stride + kernel_size - input_size)
    return padding, output_size


class FPGAModel(object):
    def __init__(self, rLUT=4e4, rThroughput=500, arch_paras=[], quan_paras=[],
                 input_shape=(3, 32, 32)):
        self.hw_paras = _get_hw_paras(arch_paras, quan_paras, input_shape)
        self.rLUT = rLUT
        self.rThroughput = rThroughput
        self.rCyc = CLOCK_FREQUENCY / self.rThroughput
        self.num_layers = len(self.hw_paras)
        self.dynamic_search()
        self.num_luts = self.partition.num_luts
        self.throughput = 0 if self.partition.num_cycs == 0 \
            else CLOCK_FREQUENCY / self.partition.num_cycs
        self.tile_pattern, self.lut_pattern, self.cyc_pattern = \
            self.partition.split()
        self.get_info()

    def search_single_layer(self, rLUT, rCyc, layer=0):
        paras = self.hw_paras[layer]
        N = paras['N']  # number of input feature maps
        M = paras['M']  # number of output feature maps
        layer = Layer(**paras)
        min_lut_layer = ()
        min_cyc_layer = ()
        for tn in range(1, N+1):
            for tm in range(1, M+1):
                num_cycs = layer.get_cycle(tn, tm)
                num_luts = layer.get_usage(tn, tm)
                # print(num_luts)
                if num_luts <= rLUT and num_cycs <= rCyc:
                    if min_lut_layer == () or num_luts < min_lut_layer[-2]:
                        min_lut_layer = (tn, tm, num_luts, num_cycs)
                    if min_cyc_layer == () or num_cycs < min_cyc_layer[-1]:
                        min_cyc_layer = (tn, tm, num_luts, num_cycs)
                    if min_lut_layer and min_cyc_layer == ():
                        print(f"inconsistent: min_lut_layer: {min_lut_layer},",
                              f" min_cyc_layer: {min_cyc_layer}")
        return min_lut_layer, min_cyc_layer

    def dynamic_search(self):
        solution_pool = [Partition()]
        for i in range(self.num_layers):
            # print(solution_pool)
            solution_pool_new = []
            for solution in solution_pool:
                min_lut_layer, min_cyc_layer = self.search_single_layer(
                    self.rLUT, self.rCyc - solution.num_cycs, i)
                if min_lut_layer:
                    solution_copy = solution.clone()
                    solution_copy.add_layer(*min_lut_layer)
                    solution_pool_new.append(solution_copy)
                if min_cyc_layer:
                    solution_copy = solution.clone()
                    solution_copy.add_layer(*min_cyc_layer)
                    solution_pool_new.append(solution_copy)
                min_lut_layer, min_cyc_layer = self.search_single_layer(
                    self.rLUT - solution.last_group_num_luts, self.rCyc - (
                        solution.num_cycs - solution.last_group_num_cycs), i)
                if min_lut_layer:
                    solution_copy = solution.clone()
                    solution_copy.append_layer(*min_lut_layer)
                    solution_pool_new.append(solution_copy)
                if min_cyc_layer:
                    solution_copy = solution.clone()
                    solution_copy.append_layer(*min_cyc_layer)
                    solution_pool_new.append(solution_copy)
            solution_pool_new = self.get_frontier(solution_pool_new)
            solution_pool = list(solution_pool_new)

        def get_all_cycs(solution):
            return solution.num_cycs
        solution_pool.sort(key=get_all_cycs)
        self.partition = solution_pool[0] if solution_pool else Partition()

    def get_frontier(self, solution_pool):
        frontier_solution_pool = []
        if solution_pool:
            def get_last_luts(solution):
                return solution.last_group_num_luts

            def get_all_cycs(solution):
                return solution.num_cycs

            def get_all_but_last_cycs(solution):
                return solution.num_cycs - solution.last_group_num_cycs

            get_metric1 = get_last_luts
            get_metric2 = get_all_cycs
            solution_pool.sort(key=get_metric1)
            frontier_solution_set = set()
            last_solution = solution_pool[0]
            for i in range(1, len(solution_pool)):
                solution = solution_pool[i]
                if get_metric2(solution) >= get_metric2(last_solution):
                    pass
                else:
                    if get_metric1(solution) == get_metric1(last_solution):
                        last_solution = solution
                    else:
                        frontier_solution_set.add(last_solution)
                        last_solution = solution
            frontier_solution_set.add(last_solution)

            get_metric1 = get_all_cycs
            get_metric2 = get_all_but_last_cycs
            solution_pool.sort(key=get_metric1)
            last_solution = solution_pool[0]
            for i in range(1, len(solution_pool)):
                solution = solution_pool[i]
                if get_metric2(solution) >= get_metric2(last_solution):
                    pass
                else:
                    if get_metric1(solution) == get_metric1(last_solution):
                        last_solution = solution
                    else:
                        frontier_solution_set.add(last_solution)
                        last_solution = solution
            frontier_solution_set.add(last_solution)

            get_metric1 = get_last_luts
            get_metric2 = get_all_but_last_cycs
            solution_pool.sort(key=get_metric1)
            last_solution = solution_pool[0]
            for i in range(1, len(solution_pool)):
                solution = solution_pool[i]
                if get_metric2(solution) >= get_metric2(last_solution):
                    pass
                else:
                    if get_metric1(solution) == get_metric1(last_solution):
                        last_solution = solution
                    else:
                        frontier_solution_set.add(last_solution)
                        last_solution = solution
            frontier_solution_set.add(last_solution)

            for solution in frontier_solution_set:
                frontier_solution_pool.append(solution)
        return frontier_solution_pool

    def get_info(self):
        return self.tile_pattern, self.lut_pattern, self.cyc_pattern, \
            self.num_luts, self.throughput

    def validate(self):
        return (self.num_luts <= self.rLUT and self.throughput >
                self.rThroughput)


class Layer(object):
    def __init__(self, **paras):
        self.N = paras['N']
        self.M = paras['M']
        self.R = paras['N']
        self.C = paras['C']
        self.Kw = paras['Kw']
        self.Kh = paras['Kh']
        self.Bii = paras['Bii']
        self.Bif = paras['Bif']
        self.Boi = paras['Boi']
        self.Bof = paras['Bof']
        self.Bwi = paras['Bwi']
        self.Bwf = paras['Bwf']

    def get_cycle(self, Tn, Tm):
        return math.ceil(self.M / Tm) * math.ceil(self.N / Tn) * \
            self.R * self.C * self.Kw * self.Kh

    def get_usage(self, Tn, Tm):
        return compute_ce_size(Tn, Tm, self.Bii, self.Bif, self.Boi, self.Bof,
                               self.Bwi, self.Bwf)


class Partition(object):
    def __init__(self):
        self.pattern = []
        self.num_layers = 0
        self.num_groups = 0
        self.num_luts = 0
        self.num_cycs = 0
        self.last_group_num_luts = 0
        self.last_group_num_cycs = 0

    def add_layer(self, tn, tm, num_luts, num_cycs):
        self.pattern.append([(tn, tm, num_luts, num_cycs)])
        self.num_layers += 1
        self.num_groups += 1
        self.num_luts = max(self.num_luts, num_luts)
        self.num_cycs += num_cycs
        self.last_group_num_luts = num_luts
        self.last_group_num_cycs = num_cycs

    def append_layer(self, tn, tm, num_luts, num_cycs):
        if self.is_empty():
            self.add_layer(tn, tm, num_luts, num_cycs)
        else:
            self.pattern[-1].append((tn, tm, num_luts, num_cycs))
            self.num_layers += 1
            self.last_group_num_luts += num_luts
            self.num_luts = max(self.num_luts, self.last_group_num_luts)
            if num_cycs > self.last_group_num_cycs:
                self.num_cycs += num_cycs - self.last_group_num_cycs
                self.last_group_num_cycs = num_cycs

    def clone(self):
        copy = Partition()
        copy.pattern = cp.deepcopy(self.pattern)
        copy.num_layers = self.num_layers
        copy.num_groups = self.num_groups
        copy.num_luts = self.num_luts
        copy.num_cycs = self.num_cycs
        copy.last_group_num_luts = self.last_group_num_luts
        copy.last_group_num_cycs = self.last_group_num_cycs
        return copy

    def is_empty(self):
        return bool(not self.pattern)

    def check(self, rLUT, rCyc):
        return (self.num_luts <= rLUT and self.num_cycs <= rCyc)

    def split(self):
        Tn_Tm = []
        num_luts = []
        num_cycs = []
        for group in self.pattern:
            group_tn_tm = []
            group_num_luts = []
            group_num_cycs = []
            for tup in group:
                group_tn_tm.append((tup[0], tup[1]))
                group_num_luts.append(tup[2])
                group_num_cycs.append(tup[3])
            Tn_Tm.append(group_tn_tm)
            num_luts.append(group_num_luts)
            num_cycs.append(group_num_cycs)
        return Tn_Tm, num_luts, num_cycs

    def __repr__(self):
        return ("pattern: {}, #LUT: {}, #CYC: {}, #LastLUT: {}, #LastCYC: {}".
                format(self.pattern, self.num_luts, self.num_cycs,
                       self.last_group_num_luts, self.last_group_num_cycs))


if __name__ == '__main__':
    arch_paras = [
        {'filter_height': 3, 'filter_width': 3, 'num_filters': 36,  # 0
         'anchor_point': []},
        {'filter_height': 3, 'filter_width': 3, 'num_filters': 48,  # 1
         'anchor_point': [1]},
        {'filter_height': 3, 'filter_width': 3, 'num_filters': 36,  # 2
         'anchor_point': [1, 1]},
        {'filter_height': 5, 'filter_width': 5, 'num_filters': 36,  # 3
         'anchor_point': [1, 1, 1]},
        {'filter_height': 3, 'filter_width': 7, 'num_filters': 48,  # 4
         'anchor_point': [0, 0, 1, 0]},
        {'filter_height': 7, 'filter_width': 7, 'num_filters': 48,  # 5
         'anchor_point': [0, 0, 0, 0, 0]}]

    quan_paras = [
    {'act_num_int_bits': 5, 'act_num_frac_bits': 7, 'weight_num_int_bits': 2, 'weight_num_frac_bits': 7},
    {'act_num_int_bits': 1, 'act_num_frac_bits': 3, 'weight_num_int_bits': 0, 'weight_num_frac_bits': 3},
    {'act_num_int_bits': 3, 'act_num_frac_bits': 3, 'weight_num_int_bits': 5, 'weight_num_frac_bits': 7},
    {'act_num_int_bits': 5, 'act_num_frac_bits': 0, 'weight_num_int_bits': 1, 'weight_num_frac_bits': 6},
    {'act_num_int_bits': 5, 'act_num_frac_bits': 4, 'weight_num_int_bits': 1, 'weight_num_frac_bits': 4},
    {'act_num_int_bits': 3, 'act_num_frac_bits': 3, 'weight_num_int_bits': 0, 'weight_num_frac_bits': 5}]

    # fpga_model = FPGAModel(
    #     arch_paras=arch_paras, quan_paras=quan_paras, rLUT=600000,
    #     rThroughput=100)
    # if fpga_model.validate():
    #     print(f"the model exists, info {fpga_model.get_info()}")
    # else:
    #     print("there doesn't exist a model that satisfies the specifications")
    #     print(list(fpga_model.get_info()))

    print(_get_hw_paras(arch_paras, quan_paras))

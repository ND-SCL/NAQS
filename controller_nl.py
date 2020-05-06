import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


input_size = 35
hidden_size = 35
num_layers = 2


class PolicyNetwork(nn.Module):
    def __init__(self, para_num_choices, para_num_layers):
        super(PolicyNetwork, self).__init__()
        self.para_num_choices = para_num_choices
        self.num_paras_per_layer = len(para_num_choices)
        self.para_num_layers = para_num_layers
        self.seq_len = self.num_paras_per_layer * self.para_num_layers
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers)
        for i in range(self.para_num_layers):
            setattr(self, 'embedding_{}_x'.format(i),
                    nn.Embedding(self.para_num_choices[-1], input_size))
            for k in range(i-1):
                setattr(self, 'classifier_{}_x{}'.format(i, k),
                        Sigmoid(hidden_size))
            for j in range(self.num_paras_per_layer):
                if j == 0:
                    setattr(self, 'embedding_{}_{}'.format(i, j),
                            nn.Embedding(2 ** i, input_size)
                            )
                else:
                    setattr(self, 'embedding_{}_{}'.format(i, j),
                            nn.Embedding(self.para_num_choices[j-1],
                                         input_size)
                            )
                setattr(self, 'classifier_{}_{}'.format(i, j),
                        nn.Linear(
                            hidden_size,
                            self.para_num_choices[j]
                            )
                        )

    def sample_para(self, x, state, layer_index=0, para_index=0):
        embedding = getattr(
            self, 'embedding_{}_{}'.format(layer_index, para_index))
        x = embedding(x)
        x, state = self.rnn(x, state)
        classifier = getattr(
            self, 'classifier_{}_{}'.format(layer_index, para_index))
        x = classifier(x)
        return x, state

    def sample_anchor(self, x, state, layer_index=0, hj_list=[]):
        embedding = getattr(self, 'embedding_{}_x'.format(layer_index))
        x = embedding(x)
        hi, state = self.rnn(x, state)
        sigmoids = []
        for k in range(len(hj_list)-1):
            classifier = getattr(
                self, 'classifier_{}_x{}'.format(layer_index, k))
            x = classifier(hj_list[k], hi)
            sigmoids.append(x)
        return sigmoids, hi, state


class Sigmoid(nn.Module):
    def __init__(self, hidden_size):
        super(Sigmoid, self).__init__()
        self.w_prev = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_curr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hj, hi):
        x = torch.tanh(self.w_prev(hj) + self.w_curr(hi))
        return torch.sigmoid(self.v(x))


class Agent():
    def __init__(self, para_space, para_num_layers, batch_size=5, lr=0.5,
                 device=torch.device('cpu')):
        self.para_space = para_space
        self.para_num_layers = para_num_layers
        self.num_paras_per_layer = len(self.para_space)
        self.para_names, self.para_values = zip(*self.para_space.items())
        self.seq_len = self.num_paras_per_layer * para_num_layers
        self.device = device
        self.batch_size = batch_size

        self.model = PolicyNetwork(tuple(len(v) for v in self.para_values),
                                   para_num_layers).to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr)
        self.optimizer = optim.Adam(self.model.parameters(), 0.005)
        self.initial_h = torch.randn(num_layers, 1, hidden_size).to(device)
        self.initial_c = torch.randn(num_layers, 1, hidden_size).to(device)
        self.initial_input = torch.randint(
            len(self.para_values[-1]), (1, 1)
            ).to(device)
        self.rollout_buffer = []
        self.reward_buffer = []
        self.reward_history = []
        self.period = 20
        self.ema = 0

    def rollout(self):
        x = self.initial_input
        state = (self.initial_h, self.initial_c)
        rollout = []
        hj_list = []
        with torch.no_grad():
            for i in range(self.para_num_layers):
                sigmoids, hi, state = self.model.sample_anchor(
                    x, state, layer_index=i, hj_list=hj_list)
                hj_list.append(hi)
                anchor_point = []
                for sig in sigmoids:
                    if random.random() < sig.item():
                        action = 1
                    else:
                        action = 0
                    anchor_point.append(action)
                # if anchor_point is []:
                #     anchor_point.append(0)
                rollout.append(anchor_point)
                x = anchor_encode(anchor_point)
                x = torch.tensor(x).expand(1, 1).to(self.device)
                for j in range(self.num_paras_per_layer):
                    x, state = self.model.sample_para(x, state, i, j)
                    pi = F.softmax(torch.squeeze(x, dim=0), dim=-1)
                    action = torch.multinomial(pi, 1)
                    x = action
                    rollout.append(action.item())
        return rollout, self._format_rollout(rollout)

    def forward(self):
        rollout_buffer = encode_rollouts(self.rollout_buffer)
        # print(rollout_buffer)
        rollout_list = [torch.tensor(v).to(self.device)
                        for v in list(zip(*rollout_buffer))]
        x = [self.initial_input.repeat(1, self.batch_size)] + \
            [rollout_list[i].unsqueeze(0) for i in range(len(rollout_list)-1)]
        state = (
            self.initial_h.repeat(1, self.batch_size, 1),
            self.initial_c.repeat(1, self.batch_size, 1)
            )
        logits = []
        hj_list = []
        for i in range(self.para_num_layers):
            sigmoids, hi, state = self.model.sample_anchor(
                x[i*(self.num_paras_per_layer+1)], state,
                layer_index=i, hj_list=hj_list)
            logits.append(sigmoids)
            hj_list.append(hi)
            for j in range(self.num_paras_per_layer):
                y, state = self.model.sample_para(
                    x[i*(self.num_paras_per_layer+1)+j+1], state, i, j)
                logits.append(F.softmax(y, dim=-1))
        return logits

    def backward(self, logits):
        rollout_list = [torch.tensor(v).to(self.device)
                        for v in list(zip(*self.rollout_buffer))]
        reward_list = \
            torch.tensor(self.reward_buffer).unsqueeze(-1).to(self.device)
        E = torch.zeros(self.batch_size, 1).to(self.device)
        for i in range(self.para_num_layers):
            sigmoids = logits[i * (self.num_paras_per_layer+1)]
            sig_actions = rollout_list[i * (self.num_paras_per_layer+1)]
            for sig, a in zip(sigmoids, sig_actions.t()):
                # print(sig.shape, sig.dtype)
                # print(a.shape, a.dtype)
                a = a.type_as(sig)
                prob = 0.5 + (-1)**a.unsqueeze(-1)*(0.5 - sig.squeeze(0))
                # print(prob.shape, prob.dtype)
                E += torch.log(prob)
                # quit()
            for j in range(self.num_paras_per_layer):
                logit = logits[i*(self.num_paras_per_layer+1)+j+1].squeeze(0)
                prob = torch.gather(logit, -1, rollout_list[
                    i*(self.num_paras_per_layer+1)+j+1].unsqueeze(-1))
                E += torch.log(prob)
        E = (E * reward_list).sum()
        if getattr(self, 'optimizer', None) is None:
            self.model.zero_grad()
            E.backward()
            with torch.no_grad():
                for p in self.model.parameters():
                    if p.grad is not None:
                        p += p.grad * lr
        else:
            loss = - E
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return -E

    def train_step(self):
        if len(self.rollout_buffer) > 0:
            logits = self.forward()
            loss = self.backward(logits)
        self.rollout_buffer.clear()
        self.reward_buffer.clear()
        self.initial_input = torch.randint_like(
            self.initial_input, len(self.para_values[-1]))
        self.initial_h = torch.randn_like(self.initial_h)
        self.initial_c = torch.randn_like(self.initial_c)
        return loss

    def store_rollout(self, rollout, reward):
        self.rollout_buffer.append(rollout)
        self.reward_buffer.append(reward-self.ema)
        self.reward_history.append(reward)
        if len(self.reward_history) > self.period:
            self.reward_history.pop(0)
        self._update_ema(reward)
        if len(self.rollout_buffer) == self.batch_size:
            self.train_step()

    def _update_ema(self, reward):
        if len(self.reward_history) < self.period:
            self.ema = np.mean(self.reward_history)
        else:
            multiplier = 2 / self.period
            self.ema = (reward - self.ema) * multiplier + self.ema

    def _format_rollout(self, actions):
        paras = []
        layer_paras = {}
        for i, v in enumerate(actions):
            # print(i, v)
            if type(v) is list:
                layer_paras['anchor_point'] = v
            else:
                para_index = i % (self.num_paras_per_layer + 1) - 1
                layer_paras[self.para_names[para_index]] = \
                    self.para_values[para_index][v]

            if (i+1) % (self.num_paras_per_layer+1) == 0:
                paras.append(layer_paras)
                layer_paras = {}
        return paras

    def adjust_learning_rate(self, lr):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr


def anchor_encode(anchor_point):
    x = 0
    for a in anchor_point:
        x = x * 2 + a
    return x


def encode_rollouts(rollout_buffer):
    out_buffer = []
    for rollout in rollout_buffer:
        encoded_rollout = []
        for action in rollout:
            if type(action) is list:
                encoded_rollout.append(anchor_encode(action))
            else:
                encoded_rollout.append(action)
        out_buffer.append(encoded_rollout)
    return out_buffer


def get_agent(para_space, para_num_layers, batch_size=5,
              device=torch.device('cpu')):
    return Agent(para_space, para_num_layers, batch_size, device)


if __name__ == '__main__':
    import torch
    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)
    from config import ARCH_SPACE, QUAN_SPACE
    from controller_bench import controller_bench
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controller_bench({**ARCH_SPACE, **QUAN_SPACE}, 6, device, skip=True, epochs=300)

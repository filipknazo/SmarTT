import os
import sys
# Ensure the project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque

torch.autograd.set_detect_anomaly(True)

import numpy as np
import random
import os


def getActionsSingle(net, current_wait_time, state):
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32)
        global_tensor = torch.tensor(current_wait_time, dtype=torch.float32)

        # Get action from the neural network
        decision_logits = net(state_tensor, global_tensor)

        # Convert logits to decision action
        temperature = 0.1
        decision_probs = F.softmax(decision_logits[0] / temperature, dim=0)
        decision_action = torch.argmax(decision_probs).item()
        #decision_action = torch.multinomial(decision_probs, num_samples=1).item()

        return decision_logits, decision_action

# Define the neural network architecture
class TrafficLightOptimizerNeuron(nn.Module):
    def __init__(self, params_tls, num_phases):
        super(TrafficLightOptimizerNeuron, self).__init__()
        hidden_size = 1
        global_params = 1
        self.global_output = 1
        self.combined_out = 4
        self.hidden_size_tcl = 1
        self.params_tls = params_tls
        self.combined_input = hidden_size + self.global_output
        self.output_possibilities = num_phases
        self.memory = deque()
        self.look_back = 10

        # Global input -> linear
        # Local input  -> lstm
        #                       -> LSTM -> linear -> ReLu -> Linear -> ReLu -> output

        self.nn_global = nn.Linear(global_params, self.global_output)
        self.nn_tcl = nn.LSTM(input_size=params_tls, hidden_size=self.hidden_size_tcl, batch_first=True)

        self.nn_combined = nn.LSTM(self.combined_input, hidden_size=self.combined_input, batch_first=True)

        # Fully connected layers
        self.nn_fc_layers = nn.Sequential(
            nn.Linear(self.combined_input, self.combined_out),
            nn.ReLU(),
            nn.Linear(self.combined_out, 2),
            nn.ReLU()
        )

        # Final output
        self.nn_output = nn.Linear(2, self.output_possibilities)

        # memory
        self.hn_c = None
        self.cn_c = None
        self.hn_tcl = None
        self.cn_tcl = None
        self.init_hidden_c()
        self.init_hidden_tcl()

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


    def forward(self, in_tcl, in_global):
        # Initialize hidden state and cell state for LSTM
        in_tcl = in_tcl.unsqueeze(0).unsqueeze(0)
        in_global = in_global.unsqueeze(0).unsqueeze(0)

        out_g = self.nn_global(in_global)
        out_tcl, (self.hn_tcl, self.cn_tcl) = self.nn_tcl(in_tcl, (self.hn_tcl, self.cn_tcl))
        out_tcl = out_tcl[:, -1, :]

        in_combined = torch.cat((out_g, out_tcl), dim=1).unsqueeze(1)
        out_combined, (self.hn_c, self.cn_c) = self.nn_combined(in_combined, (self.hn_c, self.cn_c))
        out_combined = out_combined[:, -1, :]

        out = self.nn_fc_layers(out_combined)

        return self.nn_output(out)

    def save_model(self, path):
        torch.save(self.state_dict(), f'{path}')
        # shutil.copyfile(f'{prefix}_{name_to_save}.{postfix}', f'{prefix}_{history_prefix}_{name_to_save}_{version}.{postfix}')

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def init_hidden_tcl(self):
        self.hn_tcl = torch.zeros(1, 1, self.hidden_size_tcl)
        self.cn_tcl = torch.zeros(1, 1, self.hidden_size_tcl)

    def init_hidden_c(self):
        self.hn_c = torch.zeros(1, 1, self.combined_input)
        self.cn_c = torch.zeros(1, 1, self.combined_input)


    def getLoss(sefl, loss, decision_logits, decision_action):
        decision_probs = F.softmax(decision_logits[0], dim=0)

        epsilon = 1e-10
        prob = torch.clamp(decision_probs[decision_action], min=epsilon)  # avoid log(0)

        lossAction = -torch.log(prob) * loss  # Simple policy gradient

        return lossAction


    def applyExperience(self, total_loss):
        # Backpropagation step
        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def choose_action(self, total_num_vehicles, currentWaitTime, state):
        decision_logits, decision_action = getActionsSingle(self, currentWaitTime, state)
        self.memory.append((decision_logits, decision_action, currentWaitTime, total_num_vehicles))
        return decision_action


    expected_shape = torch.Size([])
    def applyMemoryUpdates(self, total_num_vehicles, currentWaitTime):
        if len(self.memory) % 10!=0:
            return
        total_loss = 0
        for i in range(10):
            if len(self.memory) > self.look_back - 1:
                decision_logits, decision_action, previous_wait_time, vehicles = self.memory.popleft()
                loss = -(currentWaitTime-previous_wait_time)
                print(f"C:{currentWaitTime}, P:{previous_wait_time}, L:{loss}, D:{decision_action}")
                loss_tensor = self.getLoss(loss, decision_logits, decision_action)
                loss_tensor.requires_grad = True

                print(decision_action, loss)

                if loss_tensor is not None and loss_tensor.shape == self.expected_shape:
                    total_loss += loss_tensor
        self.applyExperience(loss_tensor)


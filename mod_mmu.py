from random import randint
import math
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import random
import numpy as np, sys, torch
from copy import deepcopy
import torch.nn.functional as F
from scipy.special import expit

class PT_Net(nn.Module):
    def __init__(self, num_inp, num_hid, num_mem, num_out):
        super(PT_Net, self).__init__()
        self.mmu = self.model = PT_MMU(num_inp, num_hid, num_mem)
        self.w_out = nn.Linear(num_hid, num_out)

    def forward(self, input):
        out = self.mmu.forward(input.t())
        out = F.sigmoid(self.w_out(out))
        return out.t()

    def reset(self, batch_size):
        self.mmu.reset(batch_size)


#MMU Bundle
class PT_MMU(nn.Module):
    def __init__(self, input_dim, hid_dim, mem_dim):
        super(PT_MMU, self).__init__()

        self.input_dim = input_dim; self.hid_dim = hid_dim; self.mem_dim = mem_dim

        # Input gate
        self.w_inpgate = nn.Linear(input_dim, hid_dim)
        self.w_rec_inpgate = nn.Linear(hid_dim, hid_dim)
        self.w_mem_inpgate = nn.Linear(mem_dim, hid_dim)

        # Block Input
        self.w_inp = nn.Linear(input_dim, hid_dim)
        self.w_rec_inp = nn.Linear(hid_dim, hid_dim)

        # Read Gate
        self.w_readgate = nn.Linear(input_dim, mem_dim)
        self.w_rec_readgate = nn.Linear(hid_dim, mem_dim)
        self.w_mem_readgate = nn.Linear(mem_dim, mem_dim)


        # Memory Decoder
        self.w_decoder = nn.Linear(hid_dim, mem_dim)

        # Write Gate
        self.w_writegate = nn.Linear(input_dim, mem_dim)
        self.w_rec_writegate = nn.Linear(hid_dim, mem_dim)
        self.w_mem_writegate = nn.Linear(mem_dim, mem_dim)

        # Memory Encoder
        self.w_encoder = nn.Linear(mem_dim, hid_dim)

        #Adaptive components
        self.mem = None
        self.out = None

        self.cuda()

    def reset(self, batch_size):
        # Adaptive components
        self.mem = Variable(torch.zeros(batch_size, self.mem_dim), requires_grad=1).cuda()
        self.out = Variable(torch.zeros(batch_size, self.hid_dim), requires_grad=1).cuda()


    def graph_compute(self, input, rec_output, memory):

        # Input process
        block_inp = F.elu(self.w_inp(input) + self.w_rec_inp(rec_output))  # Block Input
        inp_gate = F.sigmoid(self.w_inpgate(input) + self.w_mem_inpgate(memory) + self.w_rec_inpgate(rec_output)) #Input gate

        # Read from memory
        read_gate_out = F.sigmoid(self.w_readgate(input) + self.w_mem_readgate(memory) + self.w_rec_readgate(rec_output))
        decoded_mem = self.w_decoder(read_gate_out * memory)

        # Compute hidden activation
        hidden_act = decoded_mem + block_inp  * inp_gate

        # Update memory
        write_gate_out = F.sigmoid(self.w_writegate(input) + self.w_mem_writegate(memory) + self.w_rec_writegate(rec_output))  # #Write gate
        encoded_update = F.tanh(self.w_encoder(hidden_act))
        memory = (1 - write_gate_out) * memory + write_gate_out * encoded_update
        #memory = memory + encoded_update

        return hidden_act, memory

    def forward(self, input):
        self.out, self.mem = self.graph_compute(input, self.out, self.mem)
        return self.out

    def turn_grad_on(self):
        for param in self.parameters():
            param.requires_grad = True
            param.volatile = False

    def turn_grad_off(self):
        for param in self.parameters():
            param.requires_grad = False
            param.volatile = True





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




class PT_GRUMB(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, output_size, output_activation):
        super(PT_GRUMB, self).__init__()

        self.input_size = input_size; self.hidden_size = hidden_size; self.memory_size = memory_size; self.output_size = output_size
        if output_activation == 'sigmoid': self.output_activation = F.sigmoid
        elif output_activation == 'tanh': self.output_activation = F.tanh
        else: self.output_activation = None



        #Input gate
        self.w_inpgate = Parameter(torch.rand(memory_size, input_size), requires_grad=1)
        self.w_rec_inpgate = Parameter(torch.rand( memory_size, output_size), requires_grad=1)
        self.w_mem_inpgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        #Block Input
        self.w_inp = Parameter(torch.rand(memory_size, input_size), requires_grad=1)
        self.w_rec_inp = Parameter(torch.rand(memory_size, output_size), requires_grad=1)

        #Read Gate
        self.w_readgate = Parameter(torch.rand(memory_size, input_size), requires_grad=1)
        self.w_rec_readgate = Parameter(torch.rand(memory_size, output_size), requires_grad=1)
        self.w_mem_readgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        #Write Gate
        self.w_writegate = Parameter(torch.rand(memory_size, input_size), requires_grad=1)
        self.w_rec_writegate = Parameter(torch.rand(memory_size, output_size), requires_grad=1)
        self.w_mem_writegate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        #Output weights
        self.w_hid_out = Parameter(torch.rand(output_size, memory_size), requires_grad=1)

        #Biases
        self.w_input_gate_bias = Parameter(torch.zeros(memory_size, 1), requires_grad=1)
        self.w_block_input_bias = Parameter(torch.zeros(memory_size, 1), requires_grad=1)
        self.w_readgate_bias = Parameter(torch.zeros(memory_size, 1), requires_grad=1)
        self.w_writegate_bias = Parameter(torch.zeros(memory_size, 1), requires_grad=1)

        # Adaptive components
        self.mem = Variable(torch.zeros(self.memory_size, 1), requires_grad=1).cuda()
        self.out = Variable(torch.zeros(self.output_size, 1), requires_grad=1).cuda()

        for param in self.parameters():
            #torch.nn.init.xavier_normal(param)
            #torch.nn.init.orthogonal(param)
            #torch.nn.init.sparse(param, sparsity=0.5)
            torch.nn.init.kaiming_normal(param)

    def reset(self, batch_size):
        # Adaptive components
        self.mem = Variable(torch.zeros(self.memory_size, batch_size), requires_grad=1).cuda()
        self.out = Variable(torch.zeros(self.output_size, batch_size), requires_grad=1).cuda()

    # Some bias
    def graph_compute(self, input, rec_output, mem):
        block_inp = F.sigmoid(self.w_inp.mm(input) + self.w_rec_inp.mm(rec_output))# + self.w_block_input_bias)
        inp_gate = F.sigmoid(self.w_inpgate.mm(input) + self.w_mem_inpgate.mm(mem) + self.w_rec_inpgate.mm(
            rec_output))# + self.w_input_gate_bias)
        inp_out = block_inp * inp_gate

        mem_out = F.sigmoid(self.w_readgate.mm(input) + self.w_rec_readgate.mm(rec_output) + self.w_mem_readgate.mm(mem))# + self.w_readgate_bias) * mem

        hidden_act = mem_out + inp_out

        write_gate_out = F.sigmoid(self.w_writegate.mm(input) + self.w_mem_writegate.mm(mem) + self.w_rec_writegate.mm(rec_output))# + self.w_writegate_bias)
        mem = mem + write_gate_out * F.tanh(hidden_act)

        output = self.w_hid_out.mm(hidden_act)
        if self.output_activation != None: output = self.output_activation(output)

        return output, mem


    def forward(self, input):
        #x = Variable(input, requires_grad=True); x = x.unsqueeze(0)
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

    def to_fast_net(self):
        self.reset(1)
        keys = self.state_dict().keys()  # Get all keys
        params = self.state_dict()  # Self params
        fast_net_params = self.fast_net.param_dict  # Fast Net params

        for key in keys:
            fast_net_params[key][:] = params[key].cpu().numpy()

    def from_fast_net(self):
        keys = self.state_dict().keys() #Get all keys
        params = self.state_dict() #Self params
        fast_net_params = self.fast_net.param_dict #Fast Net params

        for key in keys:
            params[key][:] = torch.from_numpy(fast_net_params[key])

class PT_FF(nn.Module):
    def __init__(self, input_size, memory_size, output_size, output_activation):
        super(PT_FF, self).__init__()
        self.is_static = False #Distinguish between this and static policy
        self.fast_net = Fast_FF(input_size, memory_size, output_size, output_activation)
        if output_activation == 'sigmoid': self.output_activation = F.sigmoid
        elif output_activation == 'tanh': self.output_activation = F.tanh
        else: self.output_activation = None

        self.input_size = input_size; self.memory_size = memory_size; self.output_size = output_size

        #Block Input
        self.w_inp = Parameter(torch.ones(input_size, memory_size), requires_grad=1)

        #Output weights
        self.w_hid_out = Parameter(torch.rand(memory_size, output_size), requires_grad=1)

        #Adaptive components
        self.agent_sensor = 0.0; self.last_reward = 0.0

        # Turn grad off for evolutionary algorithm
        #self.turn_grad_off()


    def reset(self):
        #Adaptive components
        self.agent_sensor = 0.0; self.last_reward = 0.0

    def graph_compute(self, input):
        return F.sigmoid(input.mm(self.w_inp)).mm(self.w_hid_out)

    def forward(self, input):
        return self.graph_compute(input)

    def predict(self, input, is_static=False):
        out = self.forward(input, is_static)
        output = out.data.numpy()
        return output

    def turn_grad_on(self):
        for param in self.parameters():
            param.requires_grad = True
            param.volatile = False

    def turn_grad_off(self):
        for param in self.parameters():
            param.requires_grad = False
            param.volatile = True

    def to_fast_net(self):
        keys = self.state_dict().keys()  # Get all keys
        params = self.state_dict()  # Self params
        fast_net_params = self.fast_net.param_dict  # Fast Net params

        for key in keys:
            fast_net_params[key][:] = params[key].numpy()

    def from_fast_net(self):
        keys = self.state_dict().keys()  # Get all keys
        params = self.state_dict()  # Self params
        fast_net_params = self.fast_net.param_dict  # Fast Net params

        for key in keys:
            params[key][:] = torch.from_numpy(fast_net_params[key])

class PT_LSTM(nn.Module):
    def __init__(self,input_size, memory_size, output_size, output_activation):
        super(PT_LSTM, self).__init__()
        self.is_static = False #Distinguish between this and static policy
        self.input_size = input_size; self.memory_size = memory_size; self.output_size = output_size
        self.fast_net = Fast_LSTM(input_size, memory_size, output_size, output_activation)
        if output_activation == 'sigmoid': self.output_activation = F.sigmoid
        elif output_activation == 'tanh': self.output_activation = F.tanh
        else: self.output_activation = None

        #LSTM implementation
        # Input gate
        self.w_inpgate = Parameter(torch.rand(input_size, memory_size), requires_grad=1)
        self.w_rec_inpgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)
        self.w_mem_inpgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        # Block Input
        self.w_inp = Parameter(torch.ones(input_size, memory_size), requires_grad=1)
        self.w_rec_inp = Parameter(torch.ones(memory_size, memory_size), requires_grad=1)

        # Forget gate
        self.w_forgetgate = Parameter(torch.rand(input_size, memory_size), requires_grad=1)
        self.w_rec_forgetgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)
        self.w_mem_forgetgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        # Output gate
        self.w_outgate = Parameter(torch.rand(input_size, memory_size), requires_grad=1)
        self.w_rec_outgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)
        self.w_mem_outgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        #Hidden_to_out
        self.w_hid_out = Parameter(torch.rand(memory_size, output_size), requires_grad=1)

        #Biases
        self.w_input_gate_bias = Parameter(torch.rand(1, memory_size), requires_grad=1)
        self.w_block_input_bias = Parameter(torch.rand(1, memory_size), requires_grad=1)
        self.w_forgetgate_bias = Parameter(torch.rand(1, memory_size), requires_grad=1)
        self.w_outgate_bias = Parameter(torch.rand(1, memory_size), requires_grad=1)


        #Adaptive components
        self.c = Variable(torch.zeros(1, self.memory_size), requires_grad=1)
        self.h = Variable(torch.zeros(1, self.memory_size), requires_grad=1)

        # Turn grad off for evolutionary algorithm
        #self.turn_grad_on()

    def reset(self):
        #Adaptive components
        self.c = Variable(torch.zeros(1, self.memory_size), requires_grad=1)
        self.h = Variable(torch.zeros(1, self.memory_size), requires_grad=1)

        self.agent_sensor = 0.0; self.last_reward = 0.0

    def graph_compute(self, input):
        inp_gate = F.sigmoid(input.mm(self.w_inpgate) + self.h.mm(self.w_rec_inpgate) + self.c.mm(self.w_mem_inpgate) + self.w_input_gate_bias)
        forget_gate = F.sigmoid(input.mm(self.w_forgetgate) + self.h.mm(self.w_rec_forgetgate) + self.c.mm(self.w_mem_forgetgate) + self.w_forgetgate_bias)
        out_gate = F.sigmoid(input.mm(self.w_outgate) + self.h.mm(self.w_rec_outgate) + self.c.mm(self.w_mem_outgate) + self.w_outgate_bias)

        ct_new = F.tanh(input.mm(self.w_inp) + self.h.mm(self.w_rec_inp) + self.w_block_input_bias) #Block Input

        c_t = inp_gate * ct_new + forget_gate * self.c
        h_t = out_gate * F.tanh(c_t)
        return h_t, c_t

    def forward(self, input, is_reset):
        if is_reset: self.reset()
        self.h, self.c = self.graph_compute(input)

        return self.w_hid_out.mm(self.h)

    def predict(self, input, is_static=False):
        out = self.forward(input, is_static)
        output = out.data.numpy()
        return output

    def turn_grad_on(self):
        for param in self.parameters():
            param.requires_grad = True
            param.volatile = False

    def turn_grad_off(self):
        for param in self.parameters():
            param.requires_grad = False
            param.volatile = True


    def to_fast_net(self):
        keys = self.state_dict().keys()  # Get all keys
        params = self.state_dict()  # Self params
        fast_net_params = self.fast_net.param_dict  # Fast Net params

        for key in keys:
            fast_net_params[key][:] = params[key].numpy()


    def from_fast_net(self):
        keys = self.state_dict().keys()  # Get all keys
        params = self.state_dict()  # Self params
        fast_net_params = self.fast_net.param_dict  # Fast Net params

        for key in keys:
            params[key][:] = torch.from_numpy(fast_net_params[key])


def unpickle(filename):
    import pickle
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b

def pickle_object(obj, filename):
    with open(filename, 'wb') as output:
        cPickle.dump(obj, output, -1)


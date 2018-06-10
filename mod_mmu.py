
import torch.nn as nn
from torch.autograd import Variable
import  torch
import torch.nn.functional as F

class PT_Net(nn.Module):
    def __init__(self, num_inp, num_hid, num_mem, num_out, mem_add):
        super(PT_Net, self).__init__()
        self.mmu = self.model = PT_MMU(num_inp, num_hid, num_mem, mem_add)
        self.w_out = nn.Linear(num_hid, num_out)

    def forward(self, input):
        out = self.mmu.forward(input.t())
        out = F.sigmoid(self.w_out(out))
        return out.t()

    def reset(self, batch_size):
        self.mmu.reset(batch_size)


#MMU Bundle
class PT_MMU(nn.Module):
    def __init__(self, input_dim, hid_dim, mem_dim, mem_add):
        super(PT_MMU, self).__init__()

        self.input_dim = input_dim; self.hid_dim = hid_dim; self.mem_dim = mem_dim
        self.mem_add = mem_add

        # Input gate
        self.w_inpgate = nn.Linear(input_dim, hid_dim)
        self.w_mem_inpgate = nn.Linear(mem_dim, hid_dim)

        # Block Input
        self.w_inp = nn.Linear(input_dim, hid_dim)

        # Read Gate
        self.w_readgate = nn.Linear(input_dim, mem_dim)
        self.w_mem_readgate = nn.Linear(mem_dim, mem_dim)


        # Memory Decoder
        self.w_decoder = nn.Linear(hid_dim, mem_dim)

        # Write Gate
        self.w_writegate = nn.Linear(input_dim, mem_dim)
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


    def graph_compute(self, input, memory):

        # Input process
        block_inp = F.elu(self.w_inp(input))  # Block Input
        inp_gate = F.sigmoid(self.w_inpgate(input) + self.w_mem_inpgate(memory)) #Input gate

        # Read from memory
        read_gate_out = F.sigmoid(self.w_readgate(input) + self.w_mem_readgate(memory))
        decoded_mem = self.w_decoder(read_gate_out * memory)

        # Compute hidden activation
        hidden_act = decoded_mem + block_inp  * inp_gate

        # Update memory
        write_gate_out = F.sigmoid(self.w_writegate(input) + self.w_mem_writegate(memory))  # #Write gate
        encoded_update = F.tanh(self.w_encoder(hidden_act))
        if self.mem_add: memory = memory + encoded_update
        else: memory = (1 - write_gate_out) * memory + write_gate_out * encoded_update


        return hidden_act, memory

    def forward(self, input):
        out, self.mem = self.graph_compute(input, self.mem)
        return out

    def turn_grad_on(self):
        for param in self.parameters():
            param.requires_grad = True
            param.volatile = False

    def turn_grad_off(self):
        for param in self.parameters():
            param.requires_grad = False
            param.volatile = True





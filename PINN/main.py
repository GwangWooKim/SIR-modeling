import pandas as pd
import numpy as np

############################ data preprocessing & estimate beta, gamma ############################
####################################################################################################
df_kor = pd.read_csv('/home/users/urikokp/My/hw7/KOR.csv')
df = df_kor[['confirmed', 'recovered']].iloc[62:551,:].reset_index(drop=True)

target = df_kor['confirmed']
input = df_kor['recovered']

# collect a sample that estimates the recovry rate gamma.
a = []
for i, num in enumerate(input):
    if np.isnan(num) == False:
        if num in target.tolist():
            a.append((i-target[target==num].index).max())

# result: [16, 14, 17, 16, 19, 13, 50, 47, 20]
# gamma = 1/np.array([16, 14, 17, 16, 19, 13, 20]).mean() = 0.06
# beta = 2.2 * 0.06 = 0.132

############################ training ############################
##################################################################

import random
import torch
import torch.nn as nn
import numpy as np
import itertools
from torch.nn.parameter import Parameter
from torch.autograd import Variable

from tqdm import tqdm
import time

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

width = 50
beta = 0.132
gamma = 0.06
N = 5 * 50000000 / 50000000
I0 = 5 * 9037 / 50000000
R0 = 5 * 3507 / 50000000
S0 = N - I0 - R0

class NN(nn.Module):
    def __init__(self, width = width, N = N):
        super(NN, self).__init__()

        self.width = width
        self.N = N
        
        self.one_to_width = torch.nn.Linear(1, self.width)
        self.adaptive_activation = Parameter(torch.ones(self.width))
        self.width_to_one = torch.nn.Linear(self.width, 1)
         
    def forward(self, t, max):
        output = self.one_to_width(t) # dim : 1 -> width
        output = self.adaptive_activation * output # adaptive activation
        output = torch.tanh(output) # activation
        output = self.width_to_one(output) # dim : width -> 1
        output = torch.exp(output) # to be positive
        output = torch.clamp(output, max = max) # constraint: N = S + I + R
        return output

S_nn = NN().to(device)
I_nn = NN().to(device)
MSE = torch.nn.MSELoss()
params = [S_nn.parameters(), I_nn.parameters()]
optimizer = torch.optim.Adam(itertools.chain(*params), lr = 1e-3)

t_init = torch.zeros((1,1))
S_init = S0 * torch.ones_like(t_init).to(device)
I_init = I0 * torch.ones_like(t_init).to(device)
R_init = R0 * torch.ones_like(t_init).to(device)

loss_ = []

S_nn.train()
I_nn.train()

for epoch in tqdm(range(500)):
    optimizer.zero_grad()

    t_collocation = 489 * torch.rand((500,1))
    all_zeros = torch.zeros((500,1))

    pt_t_collocation = Variable(t_collocation, requires_grad=True).to(device)
    pt_all_zeros = Variable(all_zeros, requires_grad=False).to(device)

    t = pt_t_collocation
    s = S_nn(t, max = N)
    i = I_nn(t, max = N - s)
    r = N - s - i
    ds = torch.autograd.grad(s, t, grad_outputs=torch.ones_like(t), create_graph=True, retain_graph=True)[0]
    di = torch.autograd.grad(i, t, grad_outputs=torch.ones_like(t), create_graph=True, retain_graph=True)[0]
    dr = torch.autograd.grad(r, t, grad_outputs=torch.ones_like(t), create_graph=True, retain_graph=True)[0]
    f1 = ds + ((beta * s * i)  / N)
    f2 = di - ((beta * s * i)  / N) + (gamma * i)
    f3 = dr - (gamma * i)

    loss_1 = MSE(f1, pt_all_zeros)
    loss_2 = MSE(f2, pt_all_zeros)
    loss_3 = MSE(f3, pt_all_zeros)

    loss_f = loss_1 + loss_2 + loss_3

    pt_t_init = Variable(t_init, requires_grad=False).to(device)
    pt_I_init = Variable(I_init, requires_grad=False).to(device)
    pt_R_init = Variable(R_init, requires_grad=False).to(device)
    pt_S_init = Variable(S_init, requires_grad=False).to(device)

    init_out_1 = S_nn(pt_t_init, max = N)
    loss_1 = MSE(init_out_1, pt_S_init)

    init_out_2 = I_nn(pt_t_init, max = N - init_out_1)
    loss_2 = MSE(init_out_2, pt_I_init)

    init_out_3 = N - init_out_1 - init_out_2
    loss_3 = MSE(init_out_3, pt_R_init)

    loss_init = loss_1 + loss_2 + loss_3

    # reg1 = dr - ds
    # reg2 = 1/torch.norm(ds) + 1/torch.norm(dr)
    loss = loss_f + loss_init
    loss.backward()
    optimizer.step()

    loss_.append(loss.item())

############################ training results ############################
##########################################################################

plt.plot(loss_)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training loss')

############################ inference ############################
###################################################################

S_nn.eval()
I_nn.eval()
input = torch.range(0,489).to(device).view(-1,1)
S_list = S_nn(input, max=N).cpu().detach().numpy()
I_list = I_nn(input, max=torch.Tensor(N-S_list).to(device)).cpu().detach().numpy()
R_list = N - S_list - I_list
plt.plot(S_list, label='S')
plt.plot(I_list, label='I')
plt.plot(R_list, label='R')
plt.legend()
plt.xlabel('Time / days')
plt.ylabel('Number (/10m)')
plt.title('SIR modeling via PINN')
# plt.xlim(0,200)
# plt.ylim(0,400)

############################ comparing to ODEsolver ############################
################################################################################

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

t = np.linspace(0, 489, 489)

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number (/1000)')
# ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()
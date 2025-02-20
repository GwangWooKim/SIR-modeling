import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import itertools
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tqdm import tqdm

# Load and preprocess data
df_kor = pd.read_csv('KOR.csv')
df = df_kor[['confirmed', 'recovered']].iloc[62:551,:].reset_index(drop=True)

target = df_kor['confirmed']
input = df_kor['recovered']

# Estimate parameters
gamma = 0.06  # Recovery rate
beta = 0.132  # Transmission rate
sigma = 1/5.2  # Incubation rate (Assumed avg 5.2 days in exposed state)

# Initial conditions
N = 5 * 50000000 / 50000000
I0 = 5 * 9037 / 50000000
R0 = 5 * 3507 / 50000000
E0 = I0 * 2  # Assumption: Twice the initial infected count
S0 = N - I0 - R0 - E0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define PINN Model
class NN(nn.Module):
    def __init__(self, width=50, N=N):
        super(NN, self).__init__()
        self.width = width
        self.N = N
        self.one_to_width = torch.nn.Linear(1, self.width)
        self.adaptive_activation = Parameter(torch.ones(self.width))
        self.width_to_one = torch.nn.Linear(self.width, 1)
    
    def forward(self, t, max):
        output = self.one_to_width(t)
        output = self.adaptive_activation * output
        output = torch.tanh(output)
        output = self.width_to_one(output)
        output = torch.exp(output)
        output = torch.clamp(output, max=max)
        return output

# Instantiate models
S_nn, E_nn, I_nn = NN().to(device), NN().to(device), NN().to(device)
MSE = torch.nn.MSELoss()
params = [S_nn.parameters(), E_nn.parameters(), I_nn.parameters()]
optimizer = torch.optim.Adam(itertools.chain(*params), lr=1e-3)

# Training
loss_ = []
S_nn.train()
E_nn.train()
I_nn.train()

for epoch in tqdm(range(500)):
    optimizer.zero_grad()
    t_collocation = 489 * torch.rand((500,1))
    all_zeros = torch.zeros((500,1))
    pt_t_collocation = Variable(t_collocation, requires_grad=True).to(device)
    pt_all_zeros = Variable(all_zeros, requires_grad=False).to(device)

    t = pt_t_collocation
    s, e, i = S_nn(t, max=N), E_nn(t, max=N), I_nn(t, max=N)
    r = N - s - e - i
    ds = torch.autograd.grad(s, t, grad_outputs=torch.ones_like(t), create_graph=True)[0]
    de = torch.autograd.grad(e, t, grad_outputs=torch.ones_like(t), create_graph=True)[0]
    di = torch.autograd.grad(i, t, grad_outputs=torch.ones_like(t), create_graph=True)[0]
    dr = torch.autograd.grad(r, t, grad_outputs=torch.ones_like(t), create_graph=True)[0]

    f1 = ds + (beta * s * i) / N
    f2 = de - (beta * s * i) / N + sigma * e
    f3 = di - sigma * e + gamma * i
    f4 = dr - gamma * i

    loss_f = MSE(f1, pt_all_zeros) + MSE(f2, pt_all_zeros) + MSE(f3, pt_all_zeros) + MSE(f4, pt_all_zeros)
    loss_f.backward()
    optimizer.step()
    loss_.append(loss_f.item())

# Plot training loss
plt.plot(loss_)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training loss')
plt.show()

# ODE Solver Comparison
def deriv(y, t, N, beta, gamma, sigma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

t = np.linspace(0, 489, 489)
y0 = S0, E0, I0, R0
ret = odeint(deriv, y0, t, args=(N, beta, gamma, sigma))
S, E, I, R = ret.T

plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, E, 'y', label='Exposed')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Recovered')
plt.legend()
plt.xlabel('Time / days')
plt.ylabel('Number (/10m)')
plt.title('SEIR modeling via PINN')
plt.show()

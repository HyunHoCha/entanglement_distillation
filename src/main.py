import torch
import numpy as np
from qiskit.quantum_info import random_unitary
import torch.nn as nn
import torch.nn.functional as F
import utils
from utils import *

# python3 main.py

# CONFIG
Fid = 0.6
lr = 0.01
num_epochs = 1000

Bell_vec = torch.tensor([[1, 0, 0, 1]], dtype=torch.complex128).T / np.sqrt(2)
rest1_vec = torch.tensor([[1, 0, 0, -1]], dtype=torch.complex128).T / np.sqrt(2)
rest2_vec = torch.tensor([[0, 1, 1, 0]], dtype=torch.complex128).T / np.sqrt(2)
rest3_vec = torch.tensor([[0, 1, -1, 0]], dtype=torch.complex128).T / np.sqrt(2)

Bell_dm = Bell_vec @ Bell_vec.T.conj()
rest1_dm = rest1_vec @ rest1_vec.T.conj()
rest2_dm = rest2_vec @ rest2_vec.T.conj()
rest3_dm = rest3_vec @ rest3_vec.T.conj()

print("Specified fidelity    :", Fid)

rho_W = Bell_dm * Fid + (rest1_dm + rest2_dm + rest3_dm) * (1 - Fid) / 3
print("Check fid(rho_W, Bell):", torch.trace(rho_W @ Bell_dm).real.item())
init_dm = torch.kron(rho_W, rho_W)  # A1, B1, A2, B2


def F_new_func(U_Alice, U_Bob):
    U_Alice_4x4 = embed_1_3(U_Alice)
    U_Bob_4x4 = embed_2_4(U_Bob)
    bilateral_applied_dm = evolve_dm(evolve_dm(init_dm, U_Alice_4x4), U_Bob_4x4)
    final_dm = measure_00(bilateral_applied_dm)
    return torch.trace(final_dm @ Bell_dm).real


print("Expected F_new        :", F_new(Fid))

unitary_model = TrainableUnitary()
optimizer = torch.optim.Adam(unitary_model.parameters(), lr=lr)

for epoch in range(num_epochs):
    U_Alice, U_Bob = unitary_model()
    
    loss = -F_new_func(U_Alice, U_Bob)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(U_Alice)
print(U_Bob)

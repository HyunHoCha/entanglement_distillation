import torch
import numpy as np
from qiskit.quantum_info import random_unitary
import torch.nn as nn
import torch.nn.functional as F

# python3 main.py

# CONFIG
Fid = 0.6
lr = 0.01
num_epochs = 1000


def embed_1_3(U):
    V = torch.zeros((16, 16), dtype=torch.complex128)
    for a in [0, 1]:
        for b in [0, 1]:
            for c in [0, 1]:
                for d in [0, 1]:
                    U_index = (2 * a + b, 2 * c + d)
                    for e in [0, 1]:
                        for f in [0, 1]:
                            V_index = (8 * a + 4 * e + 2 * b + f, 8 * c + 4 * e + 2 * d + f)
                            V[V_index] = U[U_index]
    return V


def embed_2_4(U):
    V = torch.zeros((16, 16), dtype=torch.complex128)
    for a in [0, 1]:
        for b in [0, 1]:
            for c in [0, 1]:
                for d in [0, 1]:
                    U_index = (2 * a + b, 2 * c + d)
                    for e in [0, 1]:
                        for f in [0, 1]:
                            V_index = (8 * e + 4 * a + 2 * f + b, 8 * e + 4 * c + 2 * f + d)
                            V[V_index] = U[U_index]
    return V


def evolve_dm(dm, U):
    return U @ dm @ U.T.conj()


projector_00_ket = torch.zeros((4, 1), dtype=torch.complex128)
projector_00_ket[0][0] = 1
projector_00_ket = torch.kron(torch.eye(4), projector_00_ket)
projector_00_bra = projector_00_ket.T.conj()
def measure_00(rho):
    rho_proj = projector_00_bra @ rho @ projector_00_ket
    rho_proj /= torch.trace(rho_proj).real
    return rho_proj

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


def F_new(Fid):
    num = Fid ** 2 + (1 - Fid) ** 2 / 9
    den = Fid ** 2 + 2 * Fid * (1 - Fid) / 3 + 5 * (1 - Fid) ** 2 / 9
    return num / den


print("Expected F_new        :", F_new(Fid))


def F_new_func(U_Alice, U_Bob):
    U_Alice_4x4 = embed_1_3(U_Alice)
    U_Bob_4x4 = embed_2_4(U_Bob)
    bilateral_applied_dm = evolve_dm(evolve_dm(init_dm, U_Alice_4x4), U_Bob_4x4)
    final_dm = measure_00(bilateral_applied_dm)
    return torch.trace(final_dm @ Bell_dm).real


class TrainableUnitary(nn.Module):
    def __init__(self):
        super().__init__()
        self.H1 = nn.Parameter(torch.randn(4, 4, dtype=torch.cfloat))
        self.H2 = nn.Parameter(torch.randn(4, 4, dtype=torch.cfloat))
        self.H1.data = 0.5 * (self.H1.data + self.H1.data.T.conj())
        self.H2.data = 0.5 * (self.H2.data + self.H2.data.T.conj())
    
    def skew_hermitian_expm(self, H):
        iH = 1j * H
        return torch.matrix_exp(iH)

    def forward(self):
        U1 = self.skew_hermitian_expm(self.H1)
        U2 = self.skew_hermitian_expm(self.H2)
        return U1, U2


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

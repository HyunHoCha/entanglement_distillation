import torch
import numpy as np
from qiskit.quantum_info import random_unitary
import torch.nn as nn
import torch.nn.functional as F


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


def F_new(Fid):
    num = Fid ** 2 + (1 - Fid) ** 2 / 9
    den = Fid ** 2 + 2 * Fid * (1 - Fid) / 3 + 5 * (1 - Fid) ** 2 / 9
    return num / den


projector_00_ket = torch.zeros((4, 1), dtype=torch.complex128)
projector_00_ket[0][0] = 1
projector_00_ket = torch.kron(torch.eye(4), projector_00_ket)
projector_00_bra = projector_00_ket.T.conj()
def measure_00(rho):
    rho_proj = projector_00_bra @ rho @ projector_00_ket
    rho_proj /= torch.trace(rho_proj).real
    return rho_proj


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

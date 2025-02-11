import numpy as np

# python3 BBPSSW.py

# CONFIG
F = 0.6


def onehot_to_ket(onehot_1d):
    index = np.argmax(onehot_1d)
    binary_state = f"{index:04b}"
    ket = np.array([int(b) for b in binary_state])
    return ket


def ket_to_onehot(ket):
    index = int("".join(str(b) for b in ket), 2)
    onehot_1d = np.zeros(16)
    onehot_1d[index] = 1
    return onehot_1d


def embed_1_3(U):
    V = np.zeros((16, 16), dtype=complex)
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
    V = np.zeros((16, 16), dtype=complex)
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


projector_00_ket = np.zeros((4, 1), dtype=complex)
projector_00_ket[0][0] = 1
projector_00_ket = np.kron(np.eye(4), projector_00_ket)
projector_00_bra = projector_00_ket.T.conj()
def measure_00(rho):
    rho_proj = projector_00_bra @ rho @ projector_00_ket
    rho_proj /= np.trace(rho_proj).real
    return rho_proj


def F_new(F):
    num = F ** 2 + (1 - F) ** 2 / 9
    den = F ** 2 + 2 * F * (1 - F) / 3 + 5 * (1 - F) ** 2 / 9
    return num / den


Bell_vec = np.array([[1, 0, 0, 1]], dtype=complex).T / np.sqrt(2)
rest1_vec = np.array([[1, 0, 0, -1]], dtype=complex).T / np.sqrt(2)
rest2_vec = np.array([[0, 1, 1, 0]], dtype=complex).T / np.sqrt(2)
rest3_vec = np.array([[0, 1, -1, 0]], dtype=complex).T / np.sqrt(2)

Bell_dm = Bell_vec @ Bell_vec.T.conj()
rest1_dm = rest1_vec @ rest1_vec.T.conj()
rest2_dm = rest2_vec @ rest2_vec.T.conj()
rest3_dm = rest3_vec @ rest3_vec.T.conj()

CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
CNOT_1_3 = np.zeros((16, 16))
for i in range(16):
    onehot_1d = np.array([0 for _ in range(16)])
    onehot_1d[i] = 1
    ket = onehot_to_ket(onehot_1d)
    cnot_applied_ket = ket
    cnot_applied_ket[2] += cnot_applied_ket[0]
    cnot_applied_ket[2] %= 2
    cnot_applied_onehot_1d = ket_to_onehot(cnot_applied_ket)
    j = np.argmax(cnot_applied_onehot_1d)
    CNOT_1_3[j][i] = 1
CNOT_2_4 = np.zeros((16, 16))
for i in range(16):
    onehot_1d = np.array([0 for _ in range(16)])
    onehot_1d[i] = 1
    ket = onehot_to_ket(onehot_1d)
    cnot_applied_ket = ket
    cnot_applied_ket[3] += cnot_applied_ket[1]
    cnot_applied_ket[3] %= 2
    cnot_applied_onehot_1d = ket_to_onehot(cnot_applied_ket)
    j = np.argmax(cnot_applied_onehot_1d)
    CNOT_2_4[j][i] = 1

CNOT_1_3 = CNOT_1_3.astype(complex)
CNOT_2_4 = CNOT_2_4.astype(complex)

print("Specified fidelity    :", F)

rho_W = Bell_dm * F + (rest1_dm + rest2_dm + rest3_dm) * (1 - F) / 3
print("Check fid(rho_W, Bell):", np.trace(rho_W @ Bell_dm).real)
print("Expected F_new        :", F_new(F))
init_dm = np.kron(rho_W, rho_W)  # A1, B1, A2, B2

# CNOT (A1, A2)
CNOT_Alice = embed_1_3(CNOT)
# CNOT (B1, B2)
CNOT_Bob = embed_2_4(CNOT)

bilateral_applied_dm = evolve_dm(evolve_dm(init_dm, CNOT_Alice), CNOT_Bob)
final_dm = measure_00(bilateral_applied_dm)
print("True     F_new        :", np.trace(final_dm @ Bell_dm).real)

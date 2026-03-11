#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(Scalable) TEBD simulation of random quantum circuits
"""

import numpy as np
import time


CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex).reshape(
    (2, 2, 2, 2)
)


def random_u(rng):
    alpha = rng.uniform(0, 2 * np.pi)
    theta = rng.uniform(0, np.pi)
    phi = rng.uniform(0, 2 * np.pi)
    return np.exp(1j * alpha) * np.array(
        [
            [np.cos(theta / 2), -np.exp(1j * phi) * np.sin(theta / 2)],
            [np.exp(-1j * phi) * np.sin(theta / 2), np.cos(theta / 2)],
        ]
    )


def init_product_state_mps(n):
    return [np.array([[[1], [0]]], dtype=complex) for _ in range(n)]


def mps_to_state(mps):
    """
    Warning: Exponential complexity O(2^n). Use only for small n.
    """
    n = len(mps)
    state = mps[0].reshape(2, -1)
    for i in range(1, n):
        state = np.einsum("ij,jkl->ikl", state, mps[i])
        state = state.reshape(state.shape[0] * state.shape[1], state.shape[2])
    return state.reshape(-1)


def generate_single_qubit_gates(depth, n, seed=0):
    rng = np.random.default_rng(seed)
    return [[random_u(rng) for _ in range(n)] for _ in range(depth)]


def safe_inverse(lam, cutoff_val):
    inv_lam = np.zeros_like(lam)
    mask = np.abs(lam) > cutoff_val
    inv_lam[mask] = 1.0 / lam[mask]
    return inv_lam


def apply_1q_to_state(state, pos, u):
    n = state.ndim
    left = 1 << pos
    right = 1 << (n - pos - 1)
    state3 = state.reshape(left, 2, right)
    state3 = np.einsum("ij,ajb->aib", u, state3, optimize=True)
    return state3.reshape((2,) * n)


def apply_2q_to_state(state, pos, gate):
    n = state.ndim
    left = 1 << pos
    right = 1 << (n - pos - 2)
    state4 = state.reshape(left, 2, 2, right)
    state4 = np.einsum("ijst,astd->aijd", gate, state4, optimize=True)
    return state4.reshape((2,) * n)


def run_exact(n, depth, gates_1q):
    """
    厳密な状態ベクトルシミュレーション (O(2^n) コスト)
    ベンチマークおよびFidelity検証用として完全に分離
    """
    state = np.zeros((2**n, 1), dtype=complex)
    state[0] = 1
    state = state.reshape((2,) * n)

    t0 = time.perf_counter()
    for k in range(depth):
        for pos in range(n):
            U = gates_1q[k][pos]
            state = apply_1q_to_state(state, pos, U)

        for i in range(k // 2, n - 1, 2):
            state = apply_2q_to_state(state, i, CNOT)

    elapsed = time.perf_counter() - t0
    return state.reshape(-1), elapsed


def run_original(n, depth, gates_1q, max_dim=4, cutoff=1e-10, exact_state=None, verbose=False):
    mps1 = init_product_state_mps(n)

    t0 = time.perf_counter()
    for k in range(depth):
        for pos in range(n):
            U = gates_1q[k][pos]
            mps1[pos] = np.einsum("ij,kjm->kim", U, mps1[pos], optimize=True)

        for i in range(k // 2, n - 1, 2):
            pos = [i, i + 1]
            t = np.einsum(
                "ijkl,akb,blc->aijc",
                CNOT,
                mps1[pos[0]],
                mps1[pos[1]],
                optimize=True,
            )
            t = t.reshape(t.shape[0] * t.shape[1], t.shape[2] * t.shape[3])
            U1, S1, Vt1 = np.linalg.svd(t, full_matrices=False)
            S1 = S1[S1 > cutoff]
            S1 = S1[:max_dim]

            if len(S1) == 0:
                S1 = np.array([1.0])
                U1 = np.zeros((t.shape[0], 1), dtype=complex)
                U1[0, 0] = 1.0
                Vt1 = np.zeros((1, t.shape[1]), dtype=complex)
                Vt1[0, 0] = 1.0

            S1_sqrt = np.diag(np.sqrt(S1))
            mps1[pos[0]] = (U1[:, : len(S1)] @ S1_sqrt).reshape(-1, 2, len(S1))
            mps1[pos[1]] = (S1_sqrt @ Vt1[: len(S1), :]).reshape(len(S1), 2, -1)

    elapsed = time.perf_counter() - t0

    fidelity_trunc = None
    if exact_state is not None:
        state1 = mps_to_state(mps1)
        fidelity_trunc = float(np.abs(np.vdot(exact_state, state1)) ** 2)

    return {
        "elapsed": elapsed,
        "fidelity_trunc": fidelity_trunc,
        "bond_dims_trunc": [mps1[i].shape[2] for i in range(n - 1)],
    }


def vidal(n, depth, gates_1q, max_dim=4, cutoff=1e-10, exact_state=None, verbose=False):
    gammas = [np.array([[[1], [0]]], dtype=complex) for _ in range(n)]
    lambdas = [np.ones(1, dtype=float) for _ in range(n - 1)]
    unit_bond = np.ones(1, dtype=float)

    t0 = time.perf_counter()
    for k in range(depth):
        for pos in range(n):
            U = gates_1q[k][pos]
            gammas[pos] = np.einsum("ij,ajb->aib", U, gammas[pos], optimize=True)

        for i in range(k // 2, n - 1, 2):
            lam_l = lambdas[i - 1] if i > 0 else unit_bond
            lam_r = lambdas[i + 1] if (i + 1) < (n - 1) else unit_bond

            g1 = gammas[i]
            g2 = gammas[i + 1]

            theta = np.einsum(
                "a,asb,b,btd,d,ijst->aijd",
                lam_l,
                g1,
                lambdas[i],
                g2,
                lam_r,
                CNOT,
                optimize=True,
            )

            chi_l, _, _, chi_r = theta.shape
            mat = theta.reshape(chi_l * 2, 2 * chi_r)
            Uv, S, Vt = np.linalg.svd(mat, full_matrices=False)

            keep = S > cutoff
            S = S[keep]
            Uv = Uv[:, keep]
            Vt = Vt[keep, :]
            S = S[:max_dim]
            Uv = Uv[:, :max_dim]
            Vt = Vt[:max_dim, :]

            if len(S) == 0:
                S = np.array([1.0])
                Uv = np.zeros((chi_l * 2, 1), dtype=complex)
                Uv[0, 0] = 1.0
                Vt = np.zeros((1, 2 * chi_r), dtype=complex)
                Vt[0, 0] = 1.0

            r = len(S)
            lambdas[i] = S

            left = Uv.reshape(chi_l, 2, r)
            right = Vt.reshape(r, 2, chi_r)

            inv_lam_l = safe_inverse(lam_l, cutoff)
            inv_lam_r = safe_inverse(lam_r, cutoff)

            gammas[i] = left * inv_lam_l[:, None, None]
            gammas[i + 1] = right * inv_lam_r[None, None, :]

    elapsed = time.perf_counter() - t0

    fidelity = None
    if exact_state is not None:
        # 最終状態の復元とFidelity計算
        psi = gammas[0].reshape(2, gammas[0].shape[2])
        for i in range(len(lambdas)):
            psi = psi * lambdas[i][None, :]
            psi = np.einsum("ab,bcd->acd", psi, gammas[i + 1])
            psi = psi.reshape(psi.shape[0] * psi.shape[1], psi.shape[2])
        psi = psi.reshape(-1)
        fidelity = float(np.abs(np.vdot(exact_state, psi)) ** 2)

    return {
        "elapsed": elapsed,
        "fidelity": fidelity,
        "bond_dims": [len(l) for l in lambdas],
    }


def main():
    seed = 0
    cutoff = 1e-10

    print("--- Small Scale Test (n=16) with Exact State Validation ---")
    n_small = 16
    depth_small = 16
    max_dim_small = 4
    gates_1q_small = generate_single_qubit_gates(depth_small, n_small, seed=seed)

    exact_state, elapsed_exact = run_exact(n_small, depth_small, gates_1q_small)
    print(f"Exact state elapsed : {elapsed_exact:.6f} sec")

    res_orig = run_original(n_small, depth_small, gates_1q_small, max_dim_small, cutoff, exact_state=exact_state)
    res_vidal = vidal(n_small, depth_small, gates_1q_small, max_dim_small, cutoff, exact_state=exact_state)

    print(f"Original elapsed    : {res_orig['elapsed']:.6f} sec | Fidelity: {res_orig['fidelity_trunc']:.6f}")
    print(f"Vidal elapsed       : {res_vidal['elapsed']:.6f} sec | Fidelity: {res_vidal['fidelity']:.6f}")
    print(f"Speed ratio (Orig/Vidal): {res_orig['elapsed'] / res_vidal['elapsed']:.3f}\n")

    print("--- Large Scale Test (n=50) Pure TEBD Performance ---")
    n_large = 50
    depth_large = 20
    max_dim_large = 16
    gates_1q_large = generate_single_qubit_gates(depth_large, n_large, seed=seed)

    # exact_stateを渡さず、純粋なTEBDの速度のみを計測する
    res_orig_large = run_original(n_large, depth_large, gates_1q_large, max_dim_large, cutoff)
    res_vidal_large = vidal(n_large, depth_large, gates_1q_large, max_dim_large, cutoff)

    print(f"Original elapsed (n=50, max_dim=16): {res_orig_large['elapsed']:.6f} sec")
    print(f"Vidal elapsed    (n=50, max_dim=16): {res_vidal_large['elapsed']:.6f} sec")
    print(f"Speed ratio (Orig/Vidal): {res_orig_large['elapsed'] / res_vidal_large['elapsed']:.3f}")
    print(f"Final max bond dims - Original: {max(res_orig_large['bond_dims_trunc'])}, Vidal: {max(res_vidal_large['bond_dims'])}")


if __name__ == "__main__":
    main()

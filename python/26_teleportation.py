#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum Teleportation
"""

import numpy as np


def main():
    # 1-qubit basis
    zero = np.array([1, 0])

    # 演算子
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    CX = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
    ).reshape((2, 2, 2, 2))

    # 測定の射影演算子
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)

    # initial state |000>
    # 0: teleportしたいqubit (Alice)
    # 1: Bell pairのAlice側
    # 2: Bell pairのBob側
    alice = np.array([np.random.rand(), np.random.rand() * 1.0j], dtype=complex)
    alice = alice / np.linalg.norm(alice)
    state = np.kron(np.kron(alice, zero), zero).reshape((2, 2, 2))
    print(f"|Ψ> =\n{state}")
    print(f"     =\n{state.reshape((8,))}\n")

    # Bell状態を作成
    state = np.einsum(H, [1, 3], state, [0, 3, 2])
    state = np.einsum(CX, [1, 2, 4, 5], state, [0, 4, 5])

    # Aliceの操作
    state = np.einsum(CX, [0, 1, 4, 5], state, [4, 5, 2])
    state = np.einsum(H, [0, 3], state, [3, 1, 2])

    # 0番目のqubitを測定
    rho = np.einsum(state, [0, 2, 3], state.conj(), [1, 2, 3])
    if np.random.rand() < rho[0, 0].real:
        z_bit = 0
        state = np.einsum(P0, [0, 3], state, [3, 1, 2])
        state = state / np.linalg.norm(state)
    else:
        z_bit = 1
        state = np.einsum(P1, [0, 3], state, [3, 1, 2])
        state = state / np.linalg.norm(state)
    print(f"z_bit = {z_bit}")

    # 1番目のqubitを測定
    rho = np.einsum(state, [2, 0, 3], state.conj(), [2, 1, 3])
    if np.random.rand() < rho[0, 0].real:
        x_bit = 0
        state = np.einsum(P0, [1, 3], state, [0, 3, 2])
        state = state / np.linalg.norm(state)
    else:
        x_bit = 1
        state = np.einsum(P1, [1, 3], state, [0, 3, 2])
        state = state / np.linalg.norm(state)
    print(f"x_bit = {x_bit}")

    # それぞれの測定結果 (z_bit, x_bit) 応じてBobは状態を操作
    if x_bit == 1:
        state = np.einsum(X, [2, 3], state, [0, 1, 3])
    if z_bit == 1:
        state = np.einsum(Z, [2, 3], state, [0, 1, 3])

    # Aliceの状態を確認
    alice_state = np.einsum(alice, [0], alice.conj(), [1])
    print(f"Alice's state = {alice_state}")

    # Bobの状態を確認
    bob_state = np.einsum(state, [2, 3, 0], state.conj(), [2, 3, 1])
    print(f"Bob's state = {bob_state}")


if __name__ == "__main__":
    main()

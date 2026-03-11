#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum Gate Teleportation
"""

import numpy as np


def main():
    # 1-qubit basis
    zero = np.array([1, 0])

    # 演算子
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    S = np.array([[1, 0], [0, 1j]], dtype=complex)
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]).reshape(
        (2, 2, 2, 2)
    )

    # 測定の射影演算子
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)

    # input: ランダムな入力状態
    input = np.array([np.random.rand(), np.random.rand() * 1.0j], dtype=complex)
    input = input / np.linalg.norm(input)
    # ancilla: T|+>
    ancilla = np.einsum(T, [0, 1], H, [1, 2], zero, [2])
    state = np.kron(input, ancilla).reshape((2, 2))
    print(f"data state = {input}\n")
    print(f"ancilla state = {ancilla}\n")

    # CNOT(control=ancilla, target=data)
    state = np.einsum(CX, [1, 0, 3, 2], state, [2, 3])

    # 0番目のqubitを測定
    rho = np.einsum(state, [0, 2], state.conj(), [1, 2])
    if np.random.rand() < rho[0, 0].real:
        bit = 0
        state = np.einsum(P0, [0, 2], state, [2, 1])
        state = state / np.linalg.norm(state)
    else:
        bit = 1
        state = np.einsum(P1, [0, 2], state, [2, 1])
        state = state / np.linalg.norm(state)
    print(f"bit = {bit}")

    # 結果に応じてdataに補正をかける
    state = state[bit, :]
    state = state / np.linalg.norm(state)
    if bit == 1:
        state = np.einsum(X, [0, 1], state, [1])
        state = np.einsum(S, [0, 1], state, [1])
    print(f"data state after correction = {state}\n")

    print(f"T|input> = {np.einsum(T, [0, 1], input, [1])}\n")
    print(
        f"fidelity to T|input> = {np.abs(np.vdot(state, np.einsum(T, [0, 1], input, [1]))) ** 2:.9f}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Clifford conjugation tables for H, S, CNOT, CZ gates
"""

import numpy as np


def main():
    # 1. 1-qubit Pauli / Clifford gates
    Id = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    S = np.array([[1, 0], [0, 1j]], dtype=complex)

    # 2-qubit gates (basis order: |00>, |01>, |10>, |11>)
    CNOT = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
    )
    CZ = np.diag([1, 1, 1, -1]).astype(complex)

    paulis = [Id, X, Y, Z]
    pauli_labels = ["I", "X", "Y", "Z"]
    signs = [1, -1]
    sign_labels = ["+", "-"]

    # for H gate
    print("H conjugation:")
    h_table = []
    for i in range(len(paulis)):
        P = paulis[i]
        M = H @ P @ H.conj().T
        for s in range(len(signs)):
            sign = signs[s]
            for j in range(len(paulis)):
                Q = paulis[j]
                if np.allclose(M, sign * Q):
                    h_table.append((i, s, j))
                    break
    for i, s, j in h_table:
        print(f"{pauli_labels[i]} -> {sign_labels[s]} {pauli_labels[j]}")
    print()

    # for S gate
    print("S conjugation:")
    s_table = []
    for i in range(len(paulis)):
        P = paulis[i]
        M = S @ P @ S.conj().T
        for s in range(len(signs)):
            sign = signs[s]
            for j in range(len(paulis)):
                Q = paulis[j]
                if np.allclose(M, sign * Q):
                    s_table.append((i, s, j))
                    break
    for i, s, j in s_table:
        print(f"{pauli_labels[i]} -> {sign_labels[s]} {pauli_labels[j]}")
    print()

    # for CNOT gate
    print("CNOT conjugation:")
    cnot_table = []
    for i0 in range(len(paulis)):
        for i1 in range(len(paulis)):
            P = np.kron(paulis[i0], paulis[i1])
            M = CNOT @ P @ CNOT.conj().T
            for s in range(len(signs)):
                sign = signs[s]
                for j0 in range(len(paulis)):
                    for j1 in range(len(paulis)):
                        Q = np.kron(paulis[j0], paulis[j1])
                        if np.allclose(M, sign * Q):
                            cnot_table.append((i0, i1, s, j0, j1))
                            break
    for i0, i1, s, j0, j1 in cnot_table:
        print(
            f"{pauli_labels[i0]}{pauli_labels[i1]} -> {sign_labels[s]} {pauli_labels[j0]}{pauli_labels[j1]}"
        )
    print()

    # for CZ gate
    print("CZ conjugation:")
    cz_table = []
    for i0 in range(len(paulis)):
        for i1 in range(len(paulis)):
            P = np.kron(paulis[i0], paulis[i1])
            M = CZ @ P @ CZ.conj().T
            for s in range(len(signs)):
                sign = signs[s]
                for j0 in range(len(paulis)):
                    for j1 in range(len(paulis)):
                        Q = np.kron(paulis[j0], paulis[j1])
                        if np.allclose(M, sign * Q):
                            cz_table.append((i0, i1, s, j0, j1))
                            break
    for i0, i1, s, j0, j1 in cz_table:
        print(
            f"{pauli_labels[i0]}{pauli_labels[i1]} -> {sign_labels[s]} {pauli_labels[j0]}{pauli_labels[j1]}"
        )
    print()


if __name__ == "__main__":
    main()

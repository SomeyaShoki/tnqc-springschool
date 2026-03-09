#!/usr/bin/env julia
"""
Quantum Gate Teleportation
"""

using LinearAlgebra
using Printf
using Random
using TensorOperations

normalize_state(A) = A / sqrt(sum(abs2, A))

function matrix_to_tensor4(M::AbstractMatrix{T}) where {T}
    T4 = zeros(T, 2, 2, 2, 2)
    for o1 in 1:2, o2 in 1:2, i1 in 1:2, i2 in 1:2
        row = 2 * (o1 - 1) + o2
        col = 2 * (i1 - 1) + i2
        T4[o1, o2, i1, i2] = M[row, col]
    end
    return T4
end

function main()
    # 1-qubit basis
    zero = ComplexF64[1, 0]

    # 演算子
    X = ComplexF64[0 1; 1 0]
    S = ComplexF64[1 0; 0 1im]
    T = ComplexF64[1 0; 0 cis(pi / 4)]
    H = ComplexF64[1 1; 1 -1] / sqrt(2)
    CX = matrix_to_tensor4(ComplexF64[
        1 0 0 0
        0 1 0 0
        0 0 0 1
        0 0 1 0
    ])

    # 測定の射影演算子
    P0 = ComplexF64[1 0; 0 0]
    P1 = ComplexF64[0 0; 0 1]

    # input: ランダムな入力状態
    input = ComplexF64[rand(), rand() * 1im]
    input = normalize_state(input)

    # ancilla: T|+>
    @tensor ancilla[a] := T[a, b] * H[b, c] * zero[c]
    state = reshape(input, 2, 1) * transpose(reshape(ancilla, 2, 1))
    println("data state = $(input)\n")
    println("ancilla state = $(ancilla)\n")

    # CNOT(control=ancilla, target=data)
    @tensor tmp[a, b] := CX[b, a, d, c] * state[c, d]
    state = tmp

    # 0番目のqubitを測定
    state_conj = conj.(state)
    @tensor rho[a, b] := state[a, c] * state_conj[b, c]
    if rand() < real(rho[1, 1])
        bit = 0
        @tensor tmp[a, b] := P0[a, c] * state[c, b]
        state = normalize_state(tmp)
    else
        bit = 1
        @tensor tmp[a, b] := P1[a, c] * state[c, b]
        state = normalize_state(tmp)
    end
    println("bit = $bit\n")

    # 結果に応じてdataに補正をかける
    state = vec(state[bit + 1, :])
    state = normalize_state(state)
    if bit == 1
        @tensor tmp[a] := X[a, b] * state[b]
        state = tmp
        @tensor tmp[a] := S[a, b] * state[b]
        state = tmp
    end
    println("data state after correction = $(state)\n")

    @tensor target[a] := T[a, b] * input[b]
    println("T|input> = $(target)\n")
    @printf("fidelity to T|input> = %.6f\n", abs2(dot(state, target)))
end

main()

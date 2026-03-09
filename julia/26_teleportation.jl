#!/usr/bin/env julia
"""
Quantum Teleportation
"""

using Random
using TensorOperations

flatten_c_order(A) = vec(permutedims(A, collect(reverse(1:ndims(A)))))
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
    Z = ComplexF64[1 0; 0 -1]
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

    # initial state |000>
    # 0: teleportしたいqubit (Alice)
    # 1: Bell pairのAlice側
    # 2: Bell pairのBob側
    alice = ComplexF64[rand(), rand() * 1im]
    alice = normalize_state(alice)
    state = zeros(ComplexF64, 2, 2, 2)
    state[:, 1, 1] = alice
    println("|Ψ> =\n$(state)")
    println("     =\n$(flatten_c_order(state))\n")

    # Bell状態を作成
    @tensor tmp[a, b, c] := H[b, d] * state[a, d, c]
    state = tmp
    @tensor tmp[a, b, c] := CX[b, c, e, f] * state[a, e, f]
    state = tmp

    # Aliceの操作
    @tensor tmp[a, b, c] := CX[a, b, e, f] * state[e, f, c]
    state = tmp
    @tensor tmp[a, b, c] := H[a, d] * state[d, b, c]
    state = tmp

    # 0番目のqubitを測定
    state_conj = conj.(state)
    @tensor rho[a, b] := state[a, c, d] * state_conj[b, c, d]
    if rand() < real(rho[1, 1])
        z_bit = 0
        @tensor tmp[a, b, c] := P0[a, d] * state[d, b, c]
        state = normalize_state(tmp)
    else
        z_bit = 1
        @tensor tmp[a, b, c] := P1[a, d] * state[d, b, c]
        state = normalize_state(tmp)
    end
    println("z_bit = $z_bit")

    # 1番目のqubitを測定
    state_conj = conj.(state)
    @tensor rho[a, b] := state[c, a, d] * state_conj[c, b, d]
    if rand() < real(rho[1, 1])
        x_bit = 0
        @tensor tmp[a, b, c] := P0[b, d] * state[a, d, c]
        state = normalize_state(tmp)
    else
        x_bit = 1
        @tensor tmp[a, b, c] := P1[b, d] * state[a, d, c]
        state = normalize_state(tmp)
    end
    println("x_bit = $x_bit")

    # それぞれの測定結果 (z_bit, x_bit) 応じてBobは状態を操作
    if x_bit == 1
        @tensor tmp[a, b, c] := X[c, d] * state[a, b, d]
        state = tmp
    end
    if z_bit == 1
        @tensor tmp[a, b, c] := Z[c, d] * state[a, b, d]
        state = tmp
    end

    # Aliceの状態を確認
    alice_conj = conj.(alice)
    @tensor alice_state[a, b] := alice[a] * alice_conj[b]
    println("Alice's state = $(alice_state)")

    # Bobの状態を確認
    state_conj = conj.(state)
    @tensor bob_state[a, b] := state[c, d, a] * state_conj[c, d, b]
    println("Bob's state = $(bob_state)")
end

main()

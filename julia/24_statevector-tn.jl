#!/usr/bin/env julia
# Quantum circuit examples

using TensorOperations

flatten_c_order(A) = vec(permutedims(A, collect(reverse(1:ndims(A)))))

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
    # |0>
    zero = [1.0, 0.0]

    # H, CX, CZ gate
    H = [1.0 1.0; 1.0 -1.0] / sqrt(2)
    CX = matrix_to_tensor4([
        1.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0
        0.0 0.0 0.0 1.0
        0.0 0.0 1.0 0.0
    ])
    CZ = matrix_to_tensor4([
        1.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0
        0.0 0.0 1.0 0.0
        0.0 0.0 0.0 -1.0
    ])

    # initial state |000>
    state = zeros(Float64, 2, 2, 2)
    state[1, 1, 1] = 1.0
    println("|Ψ> =\n$(state)")
    println("     =\n$(flatten_c_order(state))\n")

    # apply H_0
    @tensor tmp[a, b, c] := H[a, d] * state[d, b, c]
    state = tmp
    println("|Ψ> =\n$(state)")
    println("    =\n$(flatten_c_order(state))\n")

    # apply H_2
    @tensor tmp[a, b, c] := H[c, d] * state[a, b, d]
    state = tmp
    println("|Ψ> =\n$(state)")
    println("    =\n$(flatten_c_order(state))\n")

    # apply CX_01
    @tensor tmp[a, b, c] := CX[a, b, e, f] * state[e, f, c]
    state = tmp
    println("|Ψ> =\n$(state)")
    println("    =\n$(flatten_c_order(state))\n")

    # apply CZ_12
    @tensor tmp[a, b, c] := CZ[b, c, e, f] * state[a, e, f]
    state = tmp
    println("|Ψ> =\n$(state)")
    println("    =\n$(flatten_c_order(state))\n")

    # apply H_2
    @tensor tmp[a, b, c] := H[c, d] * state[a, b, d]
    state = tmp
    println("|Ψ> =\n$(state)")
    println("    =\n$(flatten_c_order(state))\n")
end

main()

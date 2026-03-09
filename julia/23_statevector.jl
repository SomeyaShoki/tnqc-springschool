#!/usr/bin/env julia
# Quantum circuit examples

using LinearAlgebra

function main()
    # |0>
    zero = [1.0, 0.0]

    # Id, H, CX, CZ gate
    Id = [1.0 0.0; 0.0 1.0]
    H = [1.0 1.0; 1.0 -1.0] / sqrt(2)
    CX = [
        1.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0
        0.0 0.0 0.0 1.0
        0.0 0.0 1.0 0.0
    ]
    CZ = [
        1.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0
        0.0 0.0 1.0 0.0
        0.0 0.0 0.0 -1.0
    ]

    # initial state |000>
    state = kron(kron(zero, zero), zero)
    println("|Ψ> =\n$(state)\n")

    # apply H_0
    Op = kron(kron(H, Id), Id)
    state = Op * state
    println("Op =\n$(Op)\n")
    println("|Ψ> =\n$(state)\n")

    # apply H_2
    Op = kron(kron(Id, Id), H)
    state = Op * state
    println("Op =\n$(Op)\n")
    println("|Ψ> =\n$(state)\n")

    # apply CX_01
    Op = kron(CX, Id)
    state = Op * state
    println("Op =\n$(Op)\n")
    println("|Ψ> =\n$(state)\n")

    # apply CZ_12
    Op = kron(Id, CZ)
    state = Op * state
    println("Op =\n$(Op)\n")
    println("|Ψ> =\n$(state)\n")

    # apply H_2
    Op = kron(kron(Id, Id), H)
    state = Op * state
    println("Op =\n$(Op)\n")
    println("|Ψ> =\n$(state)\n")
end

main()

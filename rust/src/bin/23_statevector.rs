#[allow(unused_imports)]
use blas_src as _;
use ndarray::{Array1, Array2, array};
use std::ops::Mul;

fn kron_vec<A>(a: &Array1<A>, b: &Array1<A>) -> Array1<A>
where
    A: Copy + Mul<Output = A>,
{
    let n = b.len();
    Array1::from_shape_fn(a.len() * n, |idx| {
        let i = idx / n;
        let j = idx % n;
        a[i] * b[j]
    })
}

fn kron_mat<A>(a: &Array2<A>, b: &Array2<A>) -> Array2<A>
where
    A: Copy + Mul<Output = A>,
{
    let (ar, ac) = a.dim();
    let (br, bc) = b.dim();
    Array2::from_shape_fn((ar * br, ac * bc), |(i, j)| {
        a[[i / br, j / bc]] * b[[i % br, j % bc]]
    })
}

fn main() {
    // |0>
    let zero = array![1.0, 0.0];

    // Id, H, CX, CZ gate
    let id = array![[1.0, 0.0], [0.0, 1.0]];
    let h = array![[1.0, 1.0], [1.0, -1.0]] / 2.0_f64.sqrt();
    let cx = array![
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ];
    let cz = array![
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, -1.0]
    ];

    // initial state |000>
    let mut state = kron_vec(&kron_vec(&zero, &zero), &zero);
    println!("|Ψ> =\n{state}\n");

    // apply H_0
    let op = kron_mat(&kron_mat(&h, &id), &id);
    state = op.dot(&state);
    println!("Op =\n{op}\n");
    println!("|Ψ> =\n{state}\n");

    // apply H_2
    let op = kron_mat(&kron_mat(&id, &id), &h);
    state = op.dot(&state);
    println!("Op =\n{op}\n");
    println!("|Ψ> =\n{state}\n");

    // apply CX_01
    let op = kron_mat(&cx, &id);
    state = op.dot(&state);
    println!("Op =\n{op}\n");
    println!("|Ψ> =\n{state}\n");

    // apply CZ_12
    let op = kron_mat(&id, &cz);
    state = op.dot(&state);
    println!("Op =\n{op}\n");
    println!("|Ψ> =\n{state}\n");

    // apply H_2
    let op = kron_mat(&kron_mat(&id, &id), &h);
    state = op.dot(&state);
    println!("Op =\n{op}\n");
    println!("|Ψ> =\n{state}\n");
}

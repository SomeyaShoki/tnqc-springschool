use anyhow::Result;
use ndarray::{Array1, Array3, Array4, Ix3, array};
use ndarray_einsum::einsum;
use std::ops::Mul;
use tnqc_springschool::MapStrToAnyhowErr;

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

fn main() -> Result<()> {
    // |0>
    let zero = array![1.0, 0.0];

    // H, CX, CZ gate
    let h = array![[1.0, 1.0], [1.0, -1.0]] / 2.0_f64.sqrt();
    let cx: Array4<f64> = array![
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ]
    .into_shape_with_order((2, 2, 2, 2))?;
    let cz: Array4<f64> = array![
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, -1.0]
    ]
    .into_shape_with_order((2, 2, 2, 2))?;

    // initial state |000>
    let mut state: Array3<f64> =
        kron_vec(&kron_vec(&zero, &zero), &zero).into_shape_with_order((2, 2, 2))?;
    println!("|Ψ> =\n{state}");
    println!("     =\n{}\n", Array1::from_iter(state.iter().copied()));

    // apply H_0
    state = einsum("ad,dbc->abc", &[&h, &state])
        .map_str_err()?
        .into_dimensionality::<Ix3>()?;
    println!("|Ψ> =\n{state}");
    println!("    =\n{}\n", Array1::from_iter(state.iter().copied()));

    // apply H_2
    state = einsum("cd,abd->abc", &[&h, &state])
        .map_str_err()?
        .into_dimensionality::<Ix3>()?;
    println!("|Ψ> =\n{state}");
    println!("    =\n{}\n", Array1::from_iter(state.iter().copied()));

    // apply CX_01
    state = einsum("abef,efc->abc", &[&cx, &state])
        .map_str_err()?
        .into_dimensionality::<Ix3>()?;
    println!("|Ψ> =\n{state}");
    println!("    =\n{}\n", Array1::from_iter(state.iter().copied()));

    // apply CZ_12
    state = einsum("bcef,aef->abc", &[&cz, &state])
        .map_str_err()?
        .into_dimensionality::<Ix3>()?;
    println!("|Ψ> =\n{state}");
    println!("    =\n{}\n", Array1::from_iter(state.iter().copied()));

    // apply H_2
    state = einsum("cd,abd->abc", &[&h, &state])
        .map_str_err()?
        .into_dimensionality::<Ix3>()?;
    println!("|Ψ> =\n{state}");
    println!("    =\n{}\n", Array1::from_iter(state.iter().copied()));

    Ok(())
}

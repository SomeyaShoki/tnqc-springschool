use anyhow::Result;
use ndarray::{Array1, Array2, Array4, Ix1, Ix2, array};
use ndarray_einsum::einsum;
use num_complex::Complex64 as C64;
use rand::Rng;
use std::f64::consts::PI;
use tnqc_springschool::MapStrToAnyhowErr;

fn normalize_vec(v: &Array1<C64>) -> Array1<C64> {
    let norm = v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    v.mapv(|z| z / norm)
}

fn main() -> Result<()> {
    // 1-qubit basis
    let zero = array![C64::from(1.0), C64::from(0.0)];

    // 演算子
    let x: Array2<C64> = array![[0.0, 1.0], [1.0, 0.0]].mapv(C64::from);
    let s: Array2<C64> = array![
        [C64::from(1.0), C64::from(0.0)],
        [C64::from(0.0), C64::new(0.0, 1.0)]
    ];
    let t: Array2<C64> = array![
        [C64::from(1.0), C64::from(0.0)],
        [C64::from(0.0), C64::from_polar(1.0, PI / 4.0)]
    ];
    let h: Array2<C64> = (array![[1.0, 1.0], [1.0, -1.0]] / 2.0_f64.sqrt()).mapv(C64::from);
    let cx: Array4<C64> = array![
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ]
    .mapv(C64::from)
    .into_shape_with_order((2, 2, 2, 2))?;

    // 測定の射影演算子
    let p0: Array2<C64> = array![[1.0, 0.0], [0.0, 0.0]].mapv(C64::from);
    let p1: Array2<C64> = array![[0.0, 0.0], [0.0, 1.0]].mapv(C64::from);

    // input: ランダムな入力状態
    let mut rng = rand::rng();
    let input = array![
        C64::new(rng.random::<f64>(), 0.0),
        C64::new(0.0, rng.random::<f64>())
    ];
    let input = normalize_vec(&input);

    // ancilla: T|+>
    let ancilla: Array1<C64> = einsum("ab,bc,c->a", &[&t, &h, &zero])
        .map_str_err()?
        .into_dimensionality::<Ix1>()?;
    let mut state: Array2<C64> = einsum("a,b->ab", &[&input, &ancilla])
        .map_str_err()?
        .into_dimensionality::<Ix2>()?;
    println!("data state = {input}\n");
    println!("ancilla state = {ancilla}\n");

    // CNOT(control=ancilla, target=data)
    state = einsum("badc,cd->ab", &[&cx, &state])
        .map_str_err()?
        .into_dimensionality::<Ix2>()?;

    // 0番目のqubitを測定
    let rho: Array2<C64> = einsum("ac,bc->ab", &[&state, &state.mapv(|v| v.conj())])
        .map_str_err()?
        .into_dimensionality::<Ix2>()?;
    let bit: usize;
    if rng.random::<f64>() < rho[[0, 0]].re {
        bit = 0;
        state = einsum("ac,cb->ab", &[&p0, &state])
            .map_str_err()?
            .into_dimensionality::<Ix2>()?;
        let norm = state.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        state = state.mapv(|z| z / norm);
    } else {
        bit = 1;
        state = einsum("ac,cb->ab", &[&p1, &state])
            .map_str_err()?
            .into_dimensionality::<Ix2>()?;
        let norm = state.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        state = state.mapv(|z| z / norm);
    }
    println!("{rho} {bit}");

    // 結果に応じてdataに補正をかける
    let mut state: Array1<C64> = state.row(bit).to_owned();
    state = normalize_vec(&state);
    if bit == 1 {
        state = einsum("ab,b->a", &[&x, &state])
            .map_str_err()?
            .into_dimensionality::<Ix1>()?;
        state = einsum("ab,b->a", &[&s, &state])
            .map_str_err()?
            .into_dimensionality::<Ix1>()?;
    }
    println!("data state after correction = {state}\n");

    let target: Array1<C64> = einsum("ab,b->a", &[&t, &input])
        .map_str_err()?
        .into_dimensionality::<Ix1>()?;
    println!("T|input> = {target}\n");

    let overlap: C64 = state
        .iter()
        .zip(target.iter())
        .map(|(xv, yv)| xv.conj() * yv)
        .sum();
    println!("fidelity to T|input> = {:.6}", overlap.norm_sqr());

    Ok(())
}

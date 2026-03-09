use anyhow::Result;
use ndarray::{Array1, Array2, Array3, Array4, Ix2, Ix3, array};
use ndarray_einsum::einsum;
use num_complex::Complex64 as C64;
use rand::Rng;
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

fn normalize_vec(v: &Array1<C64>) -> Array1<C64> {
    let norm = v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    v.mapv(|z| z / norm)
}

fn normalize_state(state: &Array3<C64>) -> Array3<C64> {
    let norm = state.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    state.mapv(|z| z / norm)
}

fn main() -> Result<()> {
    // 1-qubit basis
    let zero = array![C64::from(1.0), C64::from(0.0)];

    // 演算子
    let x: Array2<C64> = array![[0.0, 1.0], [1.0, 0.0]].mapv(C64::from);
    let z: Array2<C64> = array![[1.0, 0.0], [0.0, -1.0]].mapv(C64::from);
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

    // initial state |000>
    // 0: teleportしたいqubit (Alice)
    // 1: Bell pairのAlice側
    // 2: Bell pairのBob側
    let mut rng = rand::rng();
    let alice = array![
        C64::new(rng.random::<f64>(), 0.0),
        C64::new(0.0, rng.random::<f64>())
    ];
    let alice = normalize_vec(&alice);
    let mut state: Array3<C64> =
        kron_vec(&kron_vec(&alice, &zero), &zero).into_shape_with_order((2, 2, 2))?;
    println!("|Ψ> =\n{state}");
    println!("     =\n{}\n", Array1::from_iter(state.iter().copied()));

    // Bell状態を作成
    state = einsum("bd,adc->abc", &[&h, &state])
        .map_str_err()?
        .into_dimensionality::<Ix3>()?;
    state = einsum("bcef,aef->abc", &[&cx, &state])
        .map_str_err()?
        .into_dimensionality::<Ix3>()?;

    // Aliceの操作
    state = einsum("abef,efc->abc", &[&cx, &state])
        .map_str_err()?
        .into_dimensionality::<Ix3>()?;
    state = einsum("ad,dbc->abc", &[&h, &state])
        .map_str_err()?
        .into_dimensionality::<Ix3>()?;

    // 0番目のqubitを測定
    let rho: Array2<C64> = einsum("acd,bcd->ab", &[&state, &state.mapv(|v| v.conj())])
        .map_str_err()?
        .into_dimensionality::<Ix2>()?;
    let z_bit: usize;
    if rng.random::<f64>() < rho[[0, 0]].re {
        z_bit = 0;
        state = einsum("ad,dbc->abc", &[&p0, &state])
            .map_str_err()?
            .into_dimensionality::<Ix3>()?;
        state = normalize_state(&state);
    } else {
        z_bit = 1;
        state = einsum("ad,dbc->abc", &[&p1, &state])
            .map_str_err()?
            .into_dimensionality::<Ix3>()?;
        state = normalize_state(&state);
    }
    println!("z_bit = {z_bit}");

    // 1番目のqubitを測定
    let rho: Array2<C64> = einsum("abc,adc->bd", &[&state, &state.mapv(|v| v.conj())])
        .map_str_err()?
        .into_dimensionality::<Ix2>()?;
    let x_bit: usize;
    if rng.random::<f64>() < rho[[0, 0]].re {
        x_bit = 0;
        state = einsum("bd,adc->abc", &[&p0, &state])
            .map_str_err()?
            .into_dimensionality::<Ix3>()?;
        state = normalize_state(&state);
    } else {
        x_bit = 1;
        state = einsum("bd,adc->abc", &[&p1, &state])
            .map_str_err()?
            .into_dimensionality::<Ix3>()?;
        state = normalize_state(&state);
    }
    println!("x_bit = {x_bit}");

    // それぞれの測定結果 (z_bit, x_bit) に応じてBobは状態を操作
    if x_bit == 1 {
        state = einsum("cd,abd->abc", &[&x, &state])
            .map_str_err()?
            .into_dimensionality::<Ix3>()?;
    }
    if z_bit == 1 {
        state = einsum("cd,abd->abc", &[&z, &state])
            .map_str_err()?
            .into_dimensionality::<Ix3>()?;
    }

    // Aliceの状態を確認
    let alice_state: Array2<C64> = einsum("a,b->ab", &[&alice, &alice.mapv(|v| v.conj())])
        .map_str_err()?
        .into_dimensionality::<Ix2>()?;
    println!("Alice's state = {alice_state}");

    // Bobの状態を確認
    let bob_state: Array2<C64> = einsum("abc,abd->cd", &[&state, &state.mapv(|v| v.conj())])
        .map_str_err()?
        .into_dimensionality::<Ix2>()?;
    println!("Bob's state = {bob_state}");

    Ok(())
}

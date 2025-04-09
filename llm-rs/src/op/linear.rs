use super::{Tensor, unique};
use crate::macros::*;
use digit_layout::types;
use gemm::{Parallelism::Rayon, gemm};
use mem_rearrange::Rearranging;
use std::iter::zip;

pub fn forward(y: &Tensor, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) {
    clone_tensor!(y x weight);

    let dt = unique(&[y.dt(), x.dt(), weight.dt()]).unwrap();
    assert_eq!(dt, types::F32);

    dims!([m, n] = y);
    dims!([m_, k] = x);
    dims!([n_, k_] = weight);

    assert_eq!(m, m_);
    assert_eq!(k, k_);
    assert_eq!(n, n_);

    if let Some(bias) = &bias {
        clone_tensor!(bias);

        assert_eq!(bias.dt(), dt);
        dims!([n__] = bias);
        assert_eq!(n_, n__);

        unsafe {
            Rearranging::new(
                y.layout(),
                &bias.layout().tile_be(0, &[1, n]).broadcast(0, m),
                dt.nbytes(),
            )
            .unwrap()
            .launch(y.get().write().as_mut_ptr(), bias.get().read().as_ptr())
        }
    }

    unsafe {
        gemm::<f32>(
            m,
            n,
            k,
            y.as_ref().map(|b| &mut **b.write()).mut_ptr(),
            1,
            n as isize,
            bias.is_some(),
            x.as_ref().map(|b| &**b.read()).ptr(),
            1,
            k as isize,
            weight.as_ref().map(|b| &**b.read()).ptr(),
            k as isize,
            1,
            1.,
            1.,
            false,
            false,
            false,
            Rayon(0),
        )
    }
}

pub fn backward(
    dx: &Tensor,
    dw: &Tensor,
    db: Option<&Tensor>,
    dy: &Tensor,
    x: &Tensor,
    w: &Tensor,
) {
    clone_tensor!(dx dw dy x w);

    let dt = unique(&[dx.dt(), dw.dt(), dy.dt(), x.dt(), w.dt()]).unwrap();
    assert_eq!(dt, types::F32);

    dims!([m, n] = dx);
    dims!([m_, k] = dy);
    dims!([k_, n_] = w);
    assert_eq!(m, m_);
    assert_eq!(n, n_);
    assert_eq!(k, k_);

    unsafe {
        gemm::<f32>(
            m,
            n,
            k,
            dx.as_ref().map(|b| &mut **b.write()).mut_ptr(),
            1,
            n as _,
            false,
            dy.as_ref().map(|b| &**b.read()).ptr(),
            1,
            k as _,
            w.as_ref().map(|b| &**b.read()).ptr(),
            1,
            n as _,
            1.,
            1.,
            false,
            false,
            false,
            Rayon(0),
        )
    };

    dims!([m, n] = dw);
    dims!([k, m_] = dy);
    dims!([k_, n_] = x);
    assert_eq!(m, m_);
    assert_eq!(n, n_);
    assert_eq!(k, k_);

    unsafe {
        gemm::<f32>(
            m,
            n,
            k,
            dw.as_ref().map(|b| &mut **b.write()).mut_ptr(),
            1,
            n as _,
            false,
            dy.as_ref().map(|b| &**b.read()).ptr(),
            m as _,
            1,
            x.as_ref().map(|b| &**b.read()).ptr(),
            1,
            n as _,
            1.,
            1.,
            false,
            false,
            false,
            Rayon(0),
        )
    }

    if let Some(db) = db {
        clone_tensor!(db);

        assert_eq!(db.dt(), types::F32);

        dims!([n, d] = dy);
        dims!([d_] = db);
        assert_eq!(d, d_);

        let db = db.as_ref().map(|b| &mut **b.write()).vector_mut::<f32>();
        for i in 0..n {
            let dy = dy.as_ref().index(&[i]).map(|b| &**b.read()).vector::<f32>();
            for (db, dy) in zip(&mut *db, dy) {
                *db += dy
            }
        }
    }
}

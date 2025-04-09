use super::{Tensor, gemm::mat_mul, rearrange, unique};

use crate::macros::*;
use digit_layout::types;

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

        let bias = bias.tile(0, &[1, n]).broadcast(0, m);

        rearrange::rearrange(&y, &bias);
    }

    let _ = y.as_ref().map(|b| b.release());
    mat_mul(&y, 1.0, &x, &weight.transpose(&[1, 0]), bias.map(|_| 1.0));
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

    mat_mul(&dx, 1.0, &dy, &w, None);

    dims!([m, n] = dw);
    dims!([k, m_] = dy);
    dims!([k_, n_] = x);
    assert_eq!(m, m_);
    assert_eq!(n, n_);
    assert_eq!(k, k_);

    mat_mul(&dw, 1.0, &dy.clone().transpose(&[1, 0]), &x, None);

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

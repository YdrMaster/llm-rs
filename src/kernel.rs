use crate::tensor::Tensor;
use digit_layout::types;
use gemm::{Parallelism::Rayon, gemm};
use itertools::izip;
use mem_rearrange::Rearranging;
use std::{f32::consts::PI, iter::zip, sync::LazyLock};

macro_rules! dims {
    ($pat:pat = $expr:expr) => {
        let &$pat = $expr.shape() else {
            panic!("Ndim mismatch ( = {})", $expr.shape().len())
        };
    };
}

pub fn encode(
    mut y: Tensor<&mut [u8]>,
    tok: Tensor<&[u8]>,
    wte: Tensor<&[u8]>,
    wpe: Tensor<&[u8]>,
) {
    let dt = unique(&[y.dt(), wte.dt(), wpe.dt()]).unwrap();
    assert_eq!(dt, types::F32);
    assert_eq!(tok.dt(), types::U16);

    dims!([batch_size, n_seq, d] = y);
    dims!([batch_size_, n_seq_] = tok);
    dims!([_, d_] = wte);
    dims!([_, d__] = wpe);

    assert_eq!(batch_size, batch_size_);
    assert_eq!(n_seq, n_seq_);
    unique(&[d, d_, d__]).unwrap();

    for b in 0..batch_size {
        for t in 0..n_seq {
            let out = y.as_slice_mut().index(&[b, t]).vector_mut::<f32>();
            let &ix = tok.as_slice().index(&[b, t]).scalar::<u16>();
            let wte = wte.as_slice().index(&[ix as _]).vector::<f32>();
            let wpe = wpe.as_slice().index(&[t]).vector::<f32>();
            for (out, wte, wpe) in izip!(&mut *out, wte, wpe) {
                *out = wte + wpe;
            }
        }
    }
}

pub fn layer_norm(
    mut y: Tensor<&mut [u8]>,
    mut mean: Tensor<&mut [u8]>,
    mut rstd: Tensor<&mut [u8]>,
    x: Tensor<&[u8]>,
    scalar: Tensor<&[u8]>,
    bias: Tensor<&[u8]>,
) {
    let dt = unique(&[y.dt(), mean.dt(), rstd.dt(), x.dt(), scalar.dt(), bias.dt()]).unwrap();
    assert_eq!(dt, types::F32);

    dims!([batch_size_0, n_seq_0, d_0] = y);
    dims!([batch_size_1, n_seq_1, d_1] = x);
    dims!([batch_size_2, n_seq_2] = mean);
    dims!([batch_size_3, n_seq_3] = rstd);
    dims!([d_2] = scalar);
    dims!([d_3] = bias);

    let batch_size = unique(&[batch_size_0, batch_size_1, batch_size_2, batch_size_3]).unwrap();
    let n_seq = unique(&[n_seq_0, n_seq_1, n_seq_2, n_seq_3]).unwrap();
    let d = unique(&[d_0, d_1, d_2, d_3]).unwrap();

    for b in 0..batch_size {
        for t in 0..n_seq {
            let out = y.as_slice_mut().index(&[b, t]).vector_mut::<f32>();

            let inp = x.as_slice().index(&[b, t]).vector::<f32>();
            let scalar = scalar.vector::<f32>();
            let bias = bias.vector::<f32>();

            const EPSILON: f32 = 1e-5;
            let mut sum = 0.;
            let mut sum2 = 0.;
            for &x in inp {
                sum += x;
                sum2 += x * x;
            }
            let mean_ = sum / (d as f32);
            let rstd_ = (sum2 / (d as f32) - mean_ * mean_ + EPSILON).powf(-0.5);

            *mean.as_slice_mut().index(&[b, t]).scalar_mut() = mean_;
            *rstd.as_slice_mut().index(&[b, t]).scalar_mut() = rstd_;

            for (out, scalar, bias, x) in izip!(&mut *out, scalar, bias, inp) {
                *out = (rstd_ * (x - mean_)) * scalar + bias
            }
        }
    }
}

pub fn mat_mul(
    y: Tensor<&mut [u8]>,
    x: Tensor<&[u8]>,
    weight: Tensor<&[u8]>,
    bias: Option<Tensor<&[u8]>>,
) {
    let dt = unique(&[y.dt(), x.dt(), weight.dt()]).unwrap();
    assert_eq!(dt, types::F32);

    let mut y = y.merge(0, 2);
    let x = x.merge(0, 2);

    dims!([m, n] = y);
    dims!([m_, k] = x);
    dims!([n_, k_] = weight);

    assert_eq!(m, m_);
    assert_eq!(k, k_);
    assert_eq!(n, n_);

    if let Some(bias) = &bias {
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
            .launch(y.get_mut().as_mut_ptr(), bias.get().as_ptr())
        };
    }

    unsafe {
        gemm::<f32>(
            m,
            n,
            k,
            y.mut_ptr(),
            1,
            n as isize,
            bias.is_some(),
            x.ptr(),
            1,
            k as isize,
            weight.ptr(),
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

pub fn attention(
    mut y: Tensor<&mut [u8]>,
    mut preatt: Tensor<&mut [u8]>,
    mut att: Tensor<&mut [u8]>,
    x: Tensor<&[u8]>,
) {
    let dt = unique(&[y.dt(), preatt.dt(), att.dt(), x.dt()]).unwrap();
    assert_eq!(dt, types::F32);

    dims!([batch_size_0, n_seq_0, d] = y);
    dims!([batch_size_1, n_seq_1, d3] = x);
    dims!([batch_size_2, nh_0, n_seq_2, n_seq_3] = preatt);
    dims!([batch_size_3, nh_1, n_seq_4, n_seq_5] = att);

    let batch_size = unique(&[batch_size_0, batch_size_1, batch_size_2, batch_size_3]).unwrap();
    let n_seq = unique(&[n_seq_0, n_seq_1, n_seq_2, n_seq_3, n_seq_4, n_seq_5]).unwrap();
    let nh = unique(&[nh_0, nh_1]).unwrap();
    let d = unique(&[d, d3 / 3]).unwrap();
    let dh = d / nh;
    let scale = (dh as f32).powf(-0.5);

    for b in 0..batch_size {
        let qkv = x.as_slice().index(&[b]);
        for t in 0..n_seq {
            let q = qkv.as_slice().index(&[t]).vector::<f32>();
            let y = y.as_slice_mut().index(&[b, t]).vector_mut::<f32>();

            for h in 0..nh {
                let y = &mut y[h * dh..][..dh];
                let q = &q[h * dh..][..dh];

                let preatt = preatt.as_slice_mut().index(&[b, h, t]).vector_mut::<f32>();
                let att = att.as_slice_mut().index(&[b, h, t]).vector_mut::<f32>();
                let (preatt, _) = preatt.split_at_mut(t + 1);
                let (att, tail) = att.split_at_mut(t + 1);

                // pass 1: calculate query dot key and maxval
                let mut max = f32::NEG_INFINITY;
                for (t_, val) in preatt.iter_mut().enumerate() {
                    let k = &qkv.as_slice().index(&[t_]).vector::<f32>()[d..][h * dh..][..dh];
                    *val = zip(q, k).map(|(&q, &k)| q * k).sum::<f32>() * scale;
                    if *val > max {
                        max = *val
                    }
                }

                // pass 2: calculate the exp and keep track of sum
                let mut expsum = 0.;
                for (att, preatt) in zip(&mut *att, preatt) {
                    *att = (*preatt - max).exp();
                    expsum += *att
                }
                let expsum_inv = 1. / expsum;

                // pass 3: normalize to get the softmax
                for val in &mut *att {
                    *val *= expsum_inv
                }
                tail.fill(0.);

                // pass 4: accumulate weighted values into the output of attention
                y.fill(0.);
                for (t_, val) in att.iter_mut().enumerate() {
                    let v = &qkv.as_slice().index(&[t_]).vector::<f32>()[d * 2..][h * dh..][..dh];
                    for (y, v) in zip(&mut *y, v) {
                        *y += *val * v
                    }
                }
            }
        }
    }
}

pub fn add(c: Tensor<&mut [u8]>, a: Tensor<&[u8]>, b: Tensor<&[u8]>) {
    let dt = unique(&[c.dt(), a.dt(), b.dt()]).unwrap();
    assert_eq!(dt, types::F32);

    let c = c.merge(0, 3).vector_mut::<f32>();
    let a = a.merge(0, 3).vector::<f32>();
    let b = b.merge(0, 3).vector::<f32>();
    for (c, a, b) in izip!(&mut *c, a, b) {
        *c = a + b
    }
}

const GELU_MAGIC: f32 = 0.044715;
static GELU_FACTOR: LazyLock<f32> = LazyLock::new(|| (2. / PI).sqrt());

pub fn gelu(y: Tensor<&mut [u8]>, x: Tensor<&[u8]>) {
    let dt = unique(&[y.dt(), x.dt()]).unwrap();
    assert_eq!(dt, types::F32);

    let y = y.merge(0, 3).vector_mut::<f32>();
    let x = x.merge(0, 3).vector::<f32>();
    for (y, x) in zip(y, x) {
        *y = x * 0.5 * (1. + (*GELU_FACTOR * (x + GELU_MAGIC * x.powi(3))).tanh())
    }
}

pub fn softmax(mut y: Tensor<&mut [u8]>, x: Tensor<&[u8]>, mask: usize) {
    let dt = unique(&[y.dt(), x.dt()]).unwrap();
    assert_eq!(dt, types::F32);

    dims!([batch_size, n_seq, n_voc] = y);
    dims!([batch_size_, n_seq_, n_voc_] = x);
    assert_eq!(batch_size, batch_size_);
    assert_eq!(n_seq, n_seq_);
    assert_eq!(n_voc, n_voc_);

    for b in 0..batch_size {
        for t in 0..n_seq {
            let y = y.as_slice_mut().index(&[b, t]).vector_mut::<f32>();
            let x = x.as_slice().index(&[b, t]).vector::<f32>();

            let (y, tail) = y.split_at_mut(mask);
            let x = &x[..mask];

            let max = x.iter().max_by(|a, b| f32::total_cmp(a, b)).unwrap();
            let mut expsum = 0.;
            for (y, &x) in zip(&mut *y, x) {
                *y = (x - max).exp();
                expsum += *y
            }

            for y in y {
                *y /= expsum
            }
            tail.fill(0.)
        }
    }
}

pub fn crossentropy(mut losses: Tensor<&mut [u8]>, probs: Tensor<&[u8]>, targets: Tensor<&[u8]>) {
    assert_eq!(unique(&[losses.dt(), probs.dt()]).unwrap(), types::F32);
    assert_eq!(targets.dt(), types::U16);

    dims!([batch_size_0, n_seq_0] = losses);
    dims!([batch_size_1, n_seq_1, _] = probs);
    dims!([batch_size_2, n_seq_2] = targets);

    let batch_size = unique(&[batch_size_0, batch_size_1, batch_size_2]).unwrap();
    let n_seq = unique(&[n_seq_0, n_seq_1, n_seq_2]).unwrap();

    for b in 0..batch_size {
        for t in 0..n_seq {
            let losses = losses.as_slice_mut().index(&[b, t]).scalar_mut::<f32>();
            let probs = probs.as_slice().index(&[b, t]).vector::<f32>();
            let target = targets.as_slice().index(&[b, t]).scalar::<u16>();
            *losses = -probs[*target as usize].ln()
        }
    }
}

pub mod backward {
    use super::*;
    use std::slice::from_raw_parts_mut;

    pub fn crossentropy_softmax(
        mut dlogits: Tensor<&mut [u8]>,
        dlosses: Tensor<&[u8]>,
        probs: Tensor<&[u8]>,
        targets: Tensor<&[u8]>,
    ) {
        let dt = unique(&[dlogits.dt(), dlosses.dt(), probs.dt()]).unwrap();
        assert_eq!(dt, types::F32);
        assert_eq!(targets.dt(), types::U16);

        dims!([batch_size_0, n_seq_0, n_voc_0] = dlogits);
        dims!([batch_size_1, n_seq_1] = dlosses);
        dims!([batch_size_2, n_seq_2, n_voc_1] = probs);
        dims!([batch_size_3, n_seq_3] = targets);

        let batch_size = unique(&[batch_size_0, batch_size_1, batch_size_2, batch_size_3]).unwrap();
        let n_seq = unique(&[n_seq_0, n_seq_1, n_seq_2, n_seq_3]).unwrap();
        let _ = unique(&[n_voc_0, n_voc_1]).unwrap();

        for b in 0..batch_size {
            for t in 0..n_seq {
                let dlogits = dlogits.as_slice_mut().index(&[b, t]).vector_mut::<f32>();
                let probs = probs.as_slice().index(&[b, t]).vector::<f32>();
                let dloss = *dlosses.as_slice().index(&[b, t]).scalar::<f32>();
                let ix = *targets.as_slice().index(&[b, t]).scalar::<u16>() as usize;
                for (i, (dlogit, prob)) in zip(dlogits, probs).enumerate() {
                    let indicator = if i == ix { 1. } else { 0. };
                    *dlogit += (prob - indicator) * dloss
                }
            }
        }
    }

    pub fn mat_mul(
        dx: Tensor<&mut [u8]>,
        mut dw: Tensor<&mut [u8]>,
        db: Option<Tensor<&mut [u8]>>,
        dy: Tensor<&[u8]>,
        x: Tensor<&[u8]>,
        w: Tensor<&[u8]>,
    ) {
        let dt = unique(&[dx.dt(), dw.dt(), dy.dt(), x.dt(), w.dt()]).unwrap();
        assert_eq!(dt, types::F32);

        let mut dx = dx.merge(0, 2);
        let dy = dy.merge(0, 2);

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
                dx.mut_ptr(),
                1,
                n as _,
                false,
                dy.ptr(),
                1,
                k as _,
                w.ptr(),
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

        let x = x.merge(0, 2);

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
                dw.mut_ptr(),
                1,
                n as _,
                false,
                dy.ptr(),
                m as _,
                1,
                x.ptr(),
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

        if let Some(mut db) = db {
            assert_eq!(db.dt(), types::F32);

            dims!([n, d] = dy);
            dims!([d_] = db);
            assert_eq!(d, d_);

            let db = db.vector_mut::<f32>();
            for i in 0..n {
                let dy = dy.as_slice().index(&[i]).vector::<f32>();
                for (db, dy) in zip(&mut *db, dy) {
                    *db += dy
                }
            }
        }
    }

    pub fn layer_norm(
        mut dx: Tensor<&mut [u8]>,
        mut dw: Tensor<&mut [u8]>,
        mut db: Tensor<&mut [u8]>,
        dy: Tensor<&[u8]>,
        x: Tensor<&[u8]>,
        w: Tensor<&[u8]>,
        mean: Tensor<&[u8]>,
        rstd: Tensor<&[u8]>,
    ) {
        let dt = unique(&[
            dx.dt(),
            dw.dt(),
            db.dt(),
            dy.dt(),
            x.dt(),
            w.dt(),
            mean.dt(),
            rstd.dt(),
        ])
        .unwrap();
        assert_eq!(dt, types::F32);

        dims!([batch_size_0, n_seq_0, d_0] = dx);
        dims!([batch_size_1, n_seq_1, d_1] = dy);
        dims!([batch_size_2, n_seq_2, d_2] = x);
        dims!([d_3] = dw);
        dims!([d_4] = db);
        dims!([d_5] = w);
        dims!([batch_size_3, n_seq_3] = mean);
        dims!([batch_size_4, n_seq_4] = rstd);

        let batch_size = unique(&[
            batch_size_0,
            batch_size_1,
            batch_size_2,
            batch_size_3,
            batch_size_4,
        ])
        .unwrap();
        let n_seq = unique(&[n_seq_0, n_seq_1, n_seq_2, n_seq_3, n_seq_4]).unwrap();
        let d = unique(&[d_0, d_1, d_2, d_3, d_4, d_5]).unwrap();

        for b in 0..batch_size {
            for t in 0..n_seq {
                let dx = dx.as_slice_mut().index(&[b, t]).vector_mut::<f32>();
                let dw = dw.vector_mut::<f32>();
                let db = db.vector_mut::<f32>();
                let dy = dy.as_slice().index(&[b, t]).vector::<f32>();
                let x = x.as_slice().index(&[b, t]).vector::<f32>();
                let w = w.vector::<f32>();
                let mean = *mean.as_slice().index(&[b, t]).scalar::<f32>();
                let rstd = *rstd.as_slice().index(&[b, t]).scalar::<f32>();

                let mut dnorm_mean = 0.;
                let mut dnorm_norm_mean = 0.;
                for (dy, x, w) in izip!(dy, x, w) {
                    let norm = (x - mean) * rstd;
                    let dnorm = w * dy;
                    dnorm_mean += dnorm;
                    dnorm_norm_mean += norm * dnorm;
                }
                dnorm_mean /= d as f32;
                dnorm_norm_mean /= d as f32;

                for (dx, dw, db, dy, x, w) in izip!(dx, dw, db, dy, x, w) {
                    let norm = (x - mean) * rstd;
                    let dnorm = w * dy;
                    *db += dy;
                    *dw += norm * dy;
                    *dx += rstd * (dnorm - dnorm_mean - norm * dnorm_norm_mean)
                }
            }
        }
    }

    pub fn add(da: Tensor<&mut [u8]>, db: Tensor<&mut [u8]>, dc: Tensor<&[u8]>) {
        let dt = unique(&[da.dt(), db.dt(), dc.dt()]).unwrap();
        assert_eq!(dt, types::F32);

        let da = da.merge(0, 3).vector_mut::<f32>();
        let db = db.merge(0, 3).vector_mut::<f32>();
        let dc = dc.merge(0, 3).vector::<f32>();

        for (da, db, dc) in izip!(da, db, dc) {
            *da += *dc;
            *db += *dc;
        }
    }

    pub fn gelu(dx: Tensor<&mut [u8]>, x: Tensor<&[u8]>, dy: Tensor<&[u8]>) {
        let dt = unique(&[dx.dt(), x.dt(), dy.dt()]).unwrap();
        assert_eq!(dt, types::F32);

        let dx = dx.merge(0, 3).vector_mut::<f32>();
        let x = x.merge(0, 3).vector::<f32>();
        let dy = dy.merge(0, 3).vector::<f32>();

        for (dx, x, dy) in izip!(dx, x, dy) {
            let cube = GELU_MAGIC * x.powi(3);
            let tanh_arg = *GELU_FACTOR * (x + cube);
            let tanh_val = tanh_arg.tanh();
            let cosh_val = tanh_arg.cosh();
            let sech_val = 1. / cosh_val.powi(2);
            let grad = 0.5 * (1. + tanh_val)
                + 0.5 * x * sech_val * *GELU_FACTOR * (1. + 3. * GELU_MAGIC * x.powi(2));
            *dx += *dy * grad
        }
    }

    pub fn attention(
        mut dx: Tensor<&mut [u8]>,
        mut dpreatt: Tensor<&mut [u8]>,
        mut datt: Tensor<&mut [u8]>,
        dy: Tensor<&[u8]>,
        x: Tensor<&[u8]>,
        att: Tensor<&[u8]>,
    ) {
        let dt = unique(&[dx.dt(), dpreatt.dt(), datt.dt(), dy.dt(), x.dt(), att.dt()]).unwrap();
        assert_eq!(dt, types::F32);

        dims!([batch_size_0, n_seq_0, d3_0] = dx);
        dims!([batch_size_1, nh_0, n_seq_1, n_seq_2] = dpreatt);
        dims!([batch_size_2, nh_1, n_seq_3, n_seq_4] = datt);
        dims!([batch_size_3, n_seq_5, d_0] = dy);
        dims!([batch_size_4, n_seq_6, d3_1] = x);
        dims!([batch_size_5, nh_2, n_seq_7, n_seq_8] = att);

        let batch_size = unique(&[
            batch_size_0,
            batch_size_1,
            batch_size_2,
            batch_size_3,
            batch_size_4,
            batch_size_5,
        ])
        .unwrap();
        let n_seq = unique(&[
            n_seq_0, n_seq_1, n_seq_2, n_seq_3, n_seq_4, n_seq_5, n_seq_6, n_seq_7, n_seq_8,
        ])
        .unwrap();
        let nh = unique(&[nh_0, nh_1, nh_2]).unwrap();
        let d = unique(&[d3_0 / 3, d3_1 / 3, d_0]).unwrap();

        let dh = d / nh;
        let scale = (dh as f32).powf(-0.5);

        for b in 0..batch_size {
            for t in 0..n_seq {
                for h in 0..nh {
                    let mut dx = dx.as_slice_mut().index(&[b]);
                    let x = x.as_slice().index(&[b]);

                    let dpreatt = dpreatt.as_slice_mut().index(&[b, h, t]).vector_mut::<f32>();
                    let datt = datt.as_slice_mut().index(&[b, h, t]).vector_mut::<f32>();
                    let dy = &dy.as_slice().index(&[b, t]).vector::<f32>()[h * dh..][..dh];
                    let att = att.as_slice().index(&[b, h, t]).vector::<f32>();

                    for t_ in 0..=t {
                        let dqkv = dx.as_slice_mut().index(&[t_]).vector_mut::<f32>();
                        let qkv = x.as_slice().index(&[t_]).vector::<f32>();

                        let dv = &mut dqkv[2 * d..][h * dh..][..dh];
                        let v = &qkv[2 * d..][h * dh..][..dh];
                        let datt = &mut datt[t_];
                        let att = att[t_];

                        for (dv, v, dy) in izip!(&mut *dv, v, dy) {
                            *datt += v * dy;
                            *dv += att * dy;
                        }
                    }
                    for t_ in 0..=t {
                        for t__ in 0..=t {
                            let indicator = if t_ == t__ { 1. } else { 0. };
                            dpreatt[t__] += att[t_] * (indicator - att[t__]) * datt[t_];
                        }
                    }

                    let dqkv = dx.as_slice_mut().merge(0, 2).vector_mut::<f32>();
                    let qkv = x.as_slice().merge(0, 2).vector::<f32>();

                    let dq =
                        unsafe { from_raw_parts_mut(dqkv[t * 3 * d..][h * dh..].as_mut_ptr(), dh) };
                    let q = &qkv[t * 3 * d..][h * dh..][..dh];
                    for t in 0..=t {
                        let dk = &mut dqkv[(t * 3 + 1) * d..][h * dh..][..dh];
                        let k = &qkv[(t * 3 + 1) * d..][h * dh..][..dh];
                        let dpreatt = dpreatt[t];

                        for (dq, q, dk, k) in izip!(&mut *dq, q, dk, k) {
                            *dq += k * dpreatt * scale;
                            *dk += q * dpreatt * scale;
                        }
                    }
                }
            }
        }
    }

    pub fn encode(
        mut dwte: Tensor<&mut [u8]>,
        mut dwpe: Tensor<&mut [u8]>,
        dy: Tensor<&[u8]>,
        x: Tensor<&[u8]>,
    ) {
        let dt = unique(&[dwte.dt(), dwpe.dt(), dy.dt()]).unwrap();
        assert_eq!(dt, types::F32);
        assert_eq!(x.dt(), types::U16);

        dims!([batch_size, n_seq, d] = dy);
        dims!([batch_size_, n_seq_] = x);
        dims!([_, d_] = dwte);
        dims!([_, d__] = dwpe);
        assert_eq!(batch_size, batch_size_);
        assert_eq!(n_seq, n_seq_);
        unique(&[d, d_, d__]).unwrap();

        for b in 0..batch_size {
            for t in 0..n_seq {
                let dy = dy.as_slice().index(&[b, t]).vector::<f32>();
                let ix = *x.as_slice().index(&[b, t]).scalar::<u16>() as usize;
                let dwte = dwte.as_slice_mut().index(&[ix]).vector_mut::<f32>();
                let dwpe = dwpe.as_slice_mut().index(&[t]).vector_mut::<f32>();
                for (dwte, dwpe, dy) in izip!(dwte, dwpe, dy) {
                    *dwte += dy;
                    *dwpe += dy;
                }
            }
        }
    }
}

fn unique<T: Copy + Eq>(vals: &[T]) -> Option<T> {
    let [val, tail @ ..] = vals else {
        return None;
    };
    for v in tail {
        if v != val {
            return None;
        }
    }
    Some(*val)
}

use super::{NeuralNetwork, Tensor, macros::*, unique};
use crate::Context;
use digit_layout::types;
use gemm::{Parallelism::Rayon, gemm};
use mem_rearrange::Rearranging;
use std::{iter::zip, rc::Rc};

pub struct Linear {
    w: Rc<Tensor>,
    b: Option<Rc<Tensor>>,
    x: Option<Rc<Tensor>>,
}

impl NeuralNetwork for Linear {
    type Init = (Rc<Tensor>, Option<Rc<Tensor>>);

    fn init(init: Self::Init, _ctx: &mut Context) -> Self {
        let (weight, bias) = init;
        Self {
            w: weight,
            b: bias,
            x: None,
        }
    }

    fn forward(
        &mut self,
        inputs: impl IntoIterator<Item = Rc<Tensor>>,
        ctx: &mut Context,
    ) -> Vec<Rc<Tensor>> {
        destruct!([x] = inputs);
        self.x.replace(x);
        let Self { w, b, x } = self;

        let x = x.as_deref().unwrap();
        dims!([batch_size, seq_len, _] = x);
        dims!([d, _] = w);
        let y = ctx.tensor(x.dt(), &[batch_size, seq_len, d]);

        ctx.bench(|| {
            forward(
                &y.clone().merge(0, 2),
                &x.clone().merge(0, 2),
                w,
                b.as_deref(),
            )
        });

        vec![y.share()]
    }

    fn backward(
        &mut self,
        inputs: impl IntoIterator<Item = Rc<Tensor>>,
        ctx: &mut Context,
    ) -> Vec<Rc<Tensor>> {
        destruct!([dy] = inputs);
        let Self { w, b, x } = self;

        let x = x.take().unwrap();
        let dw = ctx.write_gradient("w", w);
        let dx = ctx.tensor_zeroed(x.dt(), &x.shape());
        let db = b.as_ref().map(|b| ctx.write_gradient("b", b));
        ctx.bench(|| {
            backward(
                &dx.clone().merge(0, 2),
                &dw,
                db.as_deref(),
                &dy.cloned().merge(0, 2),
                &x.cloned().merge(0, 2),
                w,
            )
        });

        vec![dx.share()]
    }
}

fn forward(y: &Tensor, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) {
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

fn backward(dx: &Tensor, dw: &Tensor, db: Option<&Tensor>, dy: &Tensor, x: &Tensor, w: &Tensor) {
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

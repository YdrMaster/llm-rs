use super::{NeuralNetwork, Tensor};
use crate::{
    Context,
    macros::*,
    op::layer_norm::{backward, forward},
};

use std::rc::Rc;

pub struct LayerNorm {
    w: Rc<Tensor>,
    b: Rc<Tensor>,
    x: Option<Rc<Tensor>>,
    mean: Option<Tensor>,
    rstd: Option<Tensor>,
}

impl NeuralNetwork for LayerNorm {
    type Init = [Rc<Tensor>; 2];

    fn init(init: Self::Init, _ctx: &mut Context) -> Self {
        let [scalar, bias] = init;
        Self {
            w: scalar,
            b: bias,
            x: None,
            mean: None,
            rstd: None,
        }
    }

    fn forward(
        &mut self,
        inputs: impl IntoIterator<Item = Rc<Tensor>>,
        ctx: &mut Context,
    ) -> Vec<Rc<Tensor>> {
        destruct!([x] = inputs);
        self.x.replace(x);
        let Self { w, b, x, .. } = self;

        let x = x.as_ref().unwrap();
        dims!([batch_size, n_seq, d] = x);

        let y = ctx.tensor(x.dt(), &[batch_size, n_seq, d]);
        let mean = ctx.tensor(x.dt(), &[batch_size, n_seq]);
        let rstd = ctx.tensor(x.dt(), &[batch_size, n_seq]);

        ctx.bench(|| {
            forward::layer_norm(
                &y.cloned().merge(0, 2),
                &mean.cloned().merge(0, 2),
                &rstd.cloned().merge(0, 2),
                &x.cloned().merge(0, 2),
                w,
                b,
            )
        });

        self.mean.replace(mean);
        self.rstd.replace(rstd);

        vec![y.share()]
    }

    fn backward(
        &mut self,
        inputs: impl IntoIterator<Item = Rc<Tensor>>,
        ctx: &mut Context,
    ) -> Vec<Rc<Tensor>> {
        destruct!([dy] = inputs);
        let Self {
            w,
            b,
            x,
            mean,
            rstd,
        } = self;

        let x = x.take().unwrap();
        let dx = ctx.tensor_zeroed(x.dt(), &x.shape());

        let dw = ctx.write_gradient("w", w);
        let db = ctx.write_gradient("b", b);
        ctx.bench(|| {
            backward::layer_norm(
                &dx.cloned().merge(0, 2),
                &dw,
                &db,
                &dy.cloned().merge(0, 2),
                &x.cloned().merge(0, 2),
                w,
                &mean.take().unwrap().merge(0, 2),
                &rstd.take().unwrap().merge(0, 2),
            )
        });

        vec![dx.share()]
    }
}

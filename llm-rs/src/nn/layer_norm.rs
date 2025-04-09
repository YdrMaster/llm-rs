use super::{NeuralNetwork, Tensor_};
use crate::{
    TestContext,
    macros::*,
    op::layer_norm::{backward, forward},
    vm::TestVM,
};

use std::rc::Rc;

pub struct LayerNorm {
    w: Rc<Tensor_>,
    b: Rc<Tensor_>,
    x: Option<Rc<Tensor_>>,
    mean: Option<Tensor_>,
    rstd: Option<Tensor_>,
}

impl NeuralNetwork<TestVM> for LayerNorm {
    type Init = [Rc<Tensor_>; 2];

    fn init(init: Self::Init, _ctx: &mut TestContext) -> Self {
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
        inputs: impl IntoIterator<Item = Rc<Tensor_>>,
        ctx: &mut TestContext,
    ) -> Vec<Rc<Tensor_>> {
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
        inputs: impl IntoIterator<Item = Rc<Tensor_>>,
        ctx: &mut TestContext,
    ) -> Vec<Rc<Tensor_>> {
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

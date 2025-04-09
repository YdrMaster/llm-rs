use super::{NeuralNetwork, Tensor_, TestVM};
use crate::{
    TestContext,
    macros::*,
    op::attention::{backward, forward},
};
use std::rc::Rc;

pub struct Attention {
    nh: usize,
    x: Option<Rc<Tensor_>>,
    att: Option<Tensor_>,
}

impl NeuralNetwork<TestVM> for Attention {
    type Init = usize;

    fn init(init: Self::Init, _ctx: &mut TestContext) -> Self {
        Self {
            nh: init,
            x: None,
            att: None,
        }
    }

    fn forward(
        &mut self,
        inputs: impl IntoIterator<Item = Rc<Tensor_>>,
        ctx: &mut TestContext,
    ) -> Vec<Rc<Tensor_>> {
        destruct!([x] = inputs);
        self.x.replace(x);
        let Self { nh, x, .. } = self;

        let x = x.as_ref().unwrap();
        dims!([batch_size, n_seq, d3] = x);

        let d = d3 / 3;
        let y = ctx.tensor_zeroed(x.dt(), &[batch_size, n_seq, d]);
        let preatt = ctx.tensor_zeroed(x.dt(), &[batch_size, *nh, n_seq, n_seq]);
        let att = ctx.tensor_zeroed(x.dt(), &[batch_size, *nh, n_seq, n_seq]);

        ctx.bench(|| forward(&y, &preatt, &att, x));

        self.att.replace(att);

        vec![y.share()]
    }

    fn backward(
        &mut self,
        inputs: impl IntoIterator<Item = Rc<Tensor_>>,
        ctx: &mut TestContext,
    ) -> Vec<Rc<Tensor_>> {
        destruct!([dy] = inputs);
        let Self { x, att, .. } = self;

        let x = x.take().unwrap();
        let dx = ctx.tensor_zeroed(x.dt(), &x.shape());

        let att = att.take().unwrap();
        let dpreatt = ctx.tensor_zeroed(att.dt(), &att.shape());
        let datt = ctx.tensor_zeroed(att.dt(), &att.shape());

        ctx.bench(|| backward(&dx, &dpreatt, &datt, &dy, &x, &att));

        vec![dx.share()]
    }
}

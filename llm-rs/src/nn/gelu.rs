use super::{NeuralNetwork, Tensor_, TestVM};
use crate::{
    TestContext,
    macros::*,
    op::gelu::{backward, forward},
};
use std::rc::Rc;

pub struct Gelu {
    x: Option<Rc<Tensor_>>,
}

impl NeuralNetwork<TestVM> for Gelu {
    type Init = ();

    fn init(_init: Self::Init, _ctx: &mut TestContext) -> Self {
        Self { x: None }
    }

    fn forward(
        &mut self,
        inputs: impl IntoIterator<Item = Rc<Tensor_>>,
        ctx: &mut TestContext,
    ) -> Vec<Rc<Tensor_>> {
        destruct!([x] = inputs);
        self.x.replace(x);
        let Self { x } = self;

        let x = x.as_ref().unwrap();
        let y = ctx.tensor(x.dt(), &x.shape());

        ctx.bench(|| forward::gelu(&y.clone().merge(0, 2), &x.cloned().merge(0, 2)));

        vec![y.share()]
    }

    fn backward(
        &mut self,
        inputs: impl IntoIterator<Item = Rc<Tensor_>>,
        ctx: &mut TestContext,
    ) -> Vec<Rc<Tensor_>> {
        destruct!([dy] = inputs);
        let Self { x } = self;

        let x = x.take().unwrap();
        let dx = ctx.tensor_zeroed(x.dt(), &x.shape());

        ctx.bench(|| {
            backward::gelu(
                &dx.clone().merge(0, 2),
                &x.cloned().merge(0, 2),
                &dy.cloned().merge(0, 2),
            )
        });

        vec![dx.share()]
    }
}

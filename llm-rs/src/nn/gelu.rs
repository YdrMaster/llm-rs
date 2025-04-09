use super::{NeuralNetwork, Tensor, macros::*};
use crate::{
    Context,
    op::gelu::{backward, forward},
};
use std::rc::Rc;

pub struct Gelu {
    x: Option<Rc<Tensor>>,
}

impl NeuralNetwork for Gelu {
    type Init = ();

    fn init(_init: Self::Init, _ctx: &mut Context) -> Self {
        Self { x: None }
    }

    fn forward(
        &mut self,
        inputs: impl IntoIterator<Item = Rc<Tensor>>,
        ctx: &mut Context,
    ) -> Vec<Rc<Tensor>> {
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
        inputs: impl IntoIterator<Item = Rc<Tensor>>,
        ctx: &mut Context,
    ) -> Vec<Rc<Tensor>> {
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

use super::{NeuralNetwork, Tensor};
use crate::{
    Context,
    macros::*,
    op::linear::{backward, forward},
};
use std::rc::Rc;

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

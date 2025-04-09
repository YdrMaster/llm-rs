use super::{NeuralNetwork, Tensor, macros::*};
use crate::{
    Context,
    op::loss::{backward, crossentropy, softmax},
};
use std::rc::Rc;

pub struct Loss {
    n_voc: usize,
    targets: Option<Rc<Tensor>>,
    probs: Option<Tensor>,
}

impl NeuralNetwork for Loss {
    type Init = usize;

    fn init(init: Self::Init, _ctx: &mut Context) -> Self {
        Self {
            n_voc: init,
            targets: None,
            probs: None,
        }
    }

    fn forward(
        &mut self,
        inputs: impl IntoIterator<Item = Rc<Tensor>>,
        ctx: &mut Context,
    ) -> Vec<Rc<Tensor>> {
        destruct!([logits, targets] = inputs);
        self.targets.replace(targets);
        let Self {
            n_voc: nvoc,
            targets,
            ..
        } = self;

        let targets = targets.as_ref().unwrap();

        let probs = ctx.tensor(logits.dt(), &logits.shape());
        softmax(&probs, &logits, *nvoc);

        let losses = ctx.tensor(probs.dt(), &targets.shape());
        crossentropy(&losses, &probs, targets);

        self.probs.replace(probs);
        vec![losses.share()]
    }

    fn backward(
        &mut self,
        inputs: impl IntoIterator<Item = Rc<Tensor>>,
        ctx: &mut Context,
    ) -> Vec<Rc<Tensor>> {
        destruct!([dlosses] = inputs);
        let Self { targets, probs, .. } = self;

        let probs = probs.take().unwrap();
        let dlogits = ctx.tensor_zeroed(probs.dt(), &probs.shape());

        backward(&dlogits, &dlosses, &probs, &targets.take().unwrap());

        vec![dlogits.share()]
    }
}

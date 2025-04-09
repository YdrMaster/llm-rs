use super::{NeuralNetwork, Tensor};
use crate::{
    Context,
    macros::*,
    op::embedding::{BatchIter, backward, build_pos, forward},
};
use digit_layout::types;
use std::rc::Rc;

pub struct Embedding {
    te: Rc<Tensor>,
    pe: Rc<Tensor>,
    tokens: Option<Rc<Tensor>>,
}

impl NeuralNetwork for Embedding {
    type Init = [Rc<Tensor>; 2];

    fn init(init: Self::Init, _ctx: &mut Context) -> Self {
        let [te, pe] = init;
        Self {
            te,
            pe,
            tokens: None,
        }
    }

    fn forward(
        &mut self,
        inputs: impl IntoIterator<Item = Rc<Tensor>>,
        ctx: &mut Context,
    ) -> Vec<Rc<Tensor>> {
        destruct!([tokens] = inputs);
        self.tokens.replace(tokens);
        let Self { te, pe, tokens } = self;
        let tokens = tokens.as_ref().unwrap();

        dims!([batch_size, n_seq] = tokens);

        dims!([_, d] = te);
        let y = ctx.tensor(te.dt(), &[batch_size, n_seq, d]);

        let i1 = tokens.cloned().merge(0, 2);
        let mut i2 = ctx.tensor(types::U16, &[batch_size * n_seq]);
        build_pos(
            i2.get_mut().clone().write(),
            BatchIter::new(batch_size, n_seq),
        );

        ctx.bench(|| forward::embedding(&y.clone().merge(0, 2), &i1, &i2, te, pe));

        vec![y.share()]
    }

    fn backward(
        &mut self,
        inputs: impl IntoIterator<Item = Rc<Tensor>>,
        ctx: &mut Context,
    ) -> Vec<Rc<Tensor>> {
        destruct!([dy] = inputs);
        let Self { te, pe, tokens } = self;

        let dtable1 = ctx.write_gradient("wte", te);
        let dtable2 = ctx.write_gradient("wpe", pe);

        let i1 = tokens.take().unwrap();
        dims!([batch_size, n_seq] = i1);
        let mut i2 = ctx.tensor(types::U16, &[batch_size * n_seq]);
        build_pos(
            i2.get_mut().clone().write(),
            BatchIter::new(batch_size, n_seq),
        );

        ctx.bench(|| {
            backward::embedding(
                &dtable1,
                &dtable2,
                &dy.cloned().merge(0, 2),
                &i1.cloned().merge(0, 2),
                &i2,
            )
        });

        vec![]
    }
}

﻿use super::{
    NeuralNetwork, Tensor_, TestVM, embedding::Embedding, gpt2_blk::Gpt2Blk, layer_norm::LayerNorm,
    linear::Linear,
};
use crate::{Blob, TestContext, llmc};
use rw_rc::RwRc;
use std::rc::Rc;

const EMBEDDING: &str = "embedding";

#[allow(non_snake_case)]
fn BLK(i: usize) -> String {
    format!("blk[{i}]")
}

const OUTPUT_NORM: &str = "output_norm";
const LM_HEAD: &str = "lm_head";

pub struct Gpt2 {
    embedding: Embedding,
    blks: Box<[Gpt2Blk]>,
    output_norm: LayerNorm,
    lm_head: Linear,
}

impl NeuralNetwork<TestVM> for Gpt2 {
    type Init = llmc::Gpt2<RwRc<Blob>>;

    fn init(init: Self::Init, ctx: &mut TestContext) -> Self {
        let Self::Init {
            config,
            wte,
            wpe,
            blks,
            output_norm,
        } = init;

        let wte = wte.share();

        let embedding = ctx.init(EMBEDDING, [wte.clone(), wpe.share()]);
        let blks = blks
            .into_iter()
            .enumerate()
            .map(|(i, blk)| ctx.init(BLK(i), (blk, config.nh)))
            .collect();
        let output_norm = ctx.init(OUTPUT_NORM, output_norm.map(Tensor_::share));
        let lm_head = ctx.init(LM_HEAD, (wte, None));

        Self {
            embedding,
            blks,
            output_norm,
            lm_head,
        }
    }

    fn forward(
        &mut self,
        inputs: impl IntoIterator<Item = Rc<Tensor_>>,
        ctx: &mut TestContext,
    ) -> Vec<Rc<Tensor_>> {
        let Self {
            embedding,
            blks,
            output_norm,
            lm_head,
        } = self;

        let x = ctx.forward(EMBEDDING, embedding, inputs);

        let x = blks
            .iter_mut()
            .enumerate()
            .fold(x, |x, (i, blk)| ctx.forward(BLK(i), blk, x));

        let x = ctx.forward(OUTPUT_NORM, output_norm, x);
        ctx.forward(LM_HEAD, lm_head, x)
    }

    fn backward(
        &mut self,
        inputs: impl IntoIterator<Item = Rc<Tensor_>>,
        ctx: &mut TestContext,
    ) -> Vec<Rc<Tensor_>> {
        let Self {
            embedding,
            blks,
            output_norm,
            lm_head,
        } = self;

        let d = ctx.backward(LM_HEAD, lm_head, inputs);
        let d = ctx.backward(OUTPUT_NORM, output_norm, d);

        let d = blks
            .iter_mut()
            .enumerate()
            .rev()
            .fold(d, |d, (i, blk)| ctx.backward(BLK(i), blk, d));

        ctx.backward(EMBEDDING, embedding, d)
    }
}

pub mod attention;
pub mod embedding;
pub mod gelu;
pub mod gpt2;
pub mod gpt2_blk;
pub mod layer_norm;
pub mod linear;
pub mod loss;

use crate::{blob::Blob, context::Context};
use std::rc::Rc;

type Tensor = crate::Tensor<rw_rc::RwRc<Blob>>;

pub trait NeuralNetwork {
    type Init;

    fn init(init: Self::Init, ctx: &mut Context) -> Self;

    fn forward(
        &mut self,
        inputs: impl IntoIterator<Item = Rc<Tensor>>,
        ctx: &mut Context,
    ) -> Vec<Rc<Tensor>>;

    fn backward(
        &mut self,
        inputs: impl IntoIterator<Item = Rc<Tensor>>,
        ctx: &mut Context,
    ) -> Vec<Rc<Tensor>>;
}

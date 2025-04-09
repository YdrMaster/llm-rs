pub mod attention;
pub mod embedding;
pub mod gelu;
pub mod gpt2;
pub mod gpt2_blk;
pub mod layer_norm;
pub mod linear;
pub mod loss;

use crate::{
    blob::Blob,
    context::Context,
    vm::{TestVM, VirtualMachine},
};
use std::rc::Rc;

pub type Tensor_ = crate::Tensor<rw_rc::RwRc<Blob>>;

pub trait NeuralNetwork<VM: VirtualMachine> {
    type Init;

    fn init(init: Self::Init, ctx: &mut Context<VM>) -> Self;

    fn forward(
        &mut self,
        inputs: impl IntoIterator<Item = Rc<VM::Tensor>>,
        ctx: &mut Context<VM>,
    ) -> Vec<Rc<VM::Tensor>>;

    fn backward(
        &mut self,
        inputs: impl IntoIterator<Item = Rc<VM::Tensor>>,
        ctx: &mut Context<VM>,
    ) -> Vec<Rc<VM::Tensor>>;
}

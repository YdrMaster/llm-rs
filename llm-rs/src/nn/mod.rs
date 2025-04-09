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

fn unique<T: Copy + Eq>(vals: &[T]) -> Option<T> {
    let [val, tail @ ..] = vals else {
        return None;
    };
    for v in tail {
        if v != val {
            return None;
        }
    }
    Some(*val)
}

mod macros {
    macro_rules! dims {
        ($pat:pat = $tensor:expr) => {
            let &$pat = &*$tensor.shape() else {
                panic!("Ndim mismatch ( = {})", $tensor.shape().len())
            };
        };
    }

    macro_rules! strides {
        ($pat:pat = $tensor:expr) => {
            let &$pat = &*$tensor.layout().strides() else {
                panic!("Ndim mismatch ( = {})", $tensor.layout().strides().len())
            };
        };
    }

    macro_rules! destruct {
        ([$( $name:ident ),+] = $iter:expr) => {
            let mut iter = $iter.into_iter();
            $( let $name = iter.next().unwrap(); )+
            assert!(iter.next().is_none());
        };
    }

    macro_rules! clone_tensor {
        ($( $tensor:ident )+) => {
            $( let $tensor = $tensor.cloned(); )+
        };
    }

    pub(super) use {clone_tensor, destruct, dims, strides};
}

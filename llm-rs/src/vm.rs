use crate::{blob::Blob, nn::Tensor_};
use digit_layout::DigitLayout;
use rw_rc::RwRc;

pub trait VirtualMachine {
    type Tensor: Tensor;

    fn tensor(&self, dt: DigitLayout, shape: &[usize]) -> Self::Tensor;
    fn tensor_zeroed(&self, dt: DigitLayout, shape: &[usize]) -> Self::Tensor;
}

pub trait Tensor {
    fn dt(&self) -> DigitLayout;
    fn shape(&self) -> &[usize];
    fn merge(&self, axis: usize, len: usize) -> Self;
}

pub struct TestVM;

impl VirtualMachine for TestVM {
    type Tensor = crate::nn::Tensor_;

    fn tensor(&self, dt: DigitLayout, shape: &[usize]) -> Self::Tensor {
        tensor(dt, shape)
    }

    fn tensor_zeroed(&self, dt: DigitLayout, shape: &[usize]) -> Self::Tensor {
        tensor_zeroed(dt, shape)
    }
}

impl Tensor for crate::nn::Tensor_ {
    fn dt(&self) -> DigitLayout {
        self.dt()
    }

    fn shape(&self) -> &[usize] {
        self.layout().shape()
    }

    fn merge(&self, axis: usize, len: usize) -> Self {
        self.clone().merge(axis, len)
    }
}

fn tensor(dt: DigitLayout, shape: &[usize]) -> Tensor_ {
    crate::Tensor::new(dt, shape).map(Blob::new).map(RwRc::new)
}

fn tensor_zeroed(dt: DigitLayout, shape: &[usize]) -> Tensor_ {
    crate::Tensor::new(dt, shape)
        .map(Blob::new_zeroed)
        .map(RwRc::new)
}

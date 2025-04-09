use crate::{blob::Blob, nn::Tensor_};
use digit_layout::DigitLayout;
use rw_rc::RwRc;

// 这个模块定义了虚拟机和张量的trait以及相关实现
// VirtualMachine trait定义了创建张量的方法
// Tensor trait定义了对张量进行操作的方法
// 实现基于transform.rs中的方法，但做了适当调整以匹配trait的接口要求

pub trait VirtualMachine {
    type Tensor: Tensor;

    fn tensor(&self, dt: DigitLayout, shape: &[usize]) -> Self::Tensor;
    fn tensor_zeroed(&self, dt: DigitLayout, shape: &[usize]) -> Self::Tensor;
}

pub trait Tensor {
    fn dt(&self) -> DigitLayout;
    fn shape(&self) -> &[usize];
    fn merge(&self, axis: usize, len: usize) -> Self;
    fn tile(&self, axis: usize, tiles: &[usize]) -> Self;
    fn broadcast(&self, axis: usize, times: usize) -> Self;
    fn transpose(&self, perm: &[usize]) -> Self;
    fn slice(&self, axis: usize, start: usize, len: usize) -> Self;
    fn split<'a>(&'a self, axis: usize, parts: &'a [usize]) -> impl Iterator<Item = Self> + 'a;
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

    fn tile(&self, axis: usize, tiles: &[usize]) -> Self {
        self.clone().tile(axis, tiles)
    }

    fn broadcast(&self, axis: usize, times: usize) -> Self {
        self.clone().broadcast(axis, times)
    }

    fn transpose(&self, perm: &[usize]) -> Self {
        self.clone().transpose(perm)
    }

    fn slice(&self, axis: usize, start: usize, len: usize) -> Self {
        self.clone().slice(axis, start, len)
    }

    fn split<'a>(&'a self, axis: usize, parts: &'a [usize]) -> impl Iterator<Item = Self> + 'a {
        self.split(axis, parts)
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

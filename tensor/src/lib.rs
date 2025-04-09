mod fmt;
mod host;
mod transform;

use digit_layout::DigitLayout;
use ndarray_layout::{ArrayLayout, Endian::BigEndian};
use std::{
    borrow::Cow,
    ops::{Deref, DerefMut},
    rc::Rc,
};

#[derive(Clone)]
pub struct Tensor<T, const N: usize> {
    dt: DigitLayout,
    layout: ArrayLayout<N>,
    data: T,
}

impl<const N: usize> Tensor<usize, N> {
    pub fn new(dt: DigitLayout, shape: &[usize]) -> Self {
        let shape = match dt.group_size() {
            1 => Cow::Borrowed(shape),
            g => {
                let mut shape = shape.to_vec();
                let last = shape.last_mut().unwrap();
                assert_eq!(*last % g, 0);
                *last /= g;
                Cow::Owned(shape)
            }
        };

        let element_size = dt.nbytes();
        let layout = ArrayLayout::new_contiguous(&shape, BigEndian, element_size);
        let size = layout.num_elements() * element_size;
        Self {
            dt,
            layout,
            data: size,
        }
    }

    pub fn contiguous_of<U, const M: usize>(tensor: &Tensor<U, M>) -> Self {
        let dt = tensor.dt;
        let element_size = dt.nbytes();
        let layout = ArrayLayout::new_contiguous(tensor.layout.shape(), BigEndian, element_size);
        let size = layout.num_elements() * element_size;
        Self {
            dt,
            layout,
            data: size,
        }
    }
}

impl<T, const N: usize> Tensor<T, N> {
    pub fn share(self) -> Rc<Self> {
        self.into()
    }

    pub const fn dt(&self) -> DigitLayout {
        self.dt
    }

    pub const fn layout(&self) -> &ArrayLayout<N> {
        &self.layout
    }

    pub fn take(self) -> T {
        self.data
    }

    pub const fn get(&self) -> &T {
        &self.data
    }

    pub fn get_mut(&mut self) -> &mut T {
        &mut self.data
    }

    pub fn shape(&self) -> Cow<[usize]> {
        match self.dt.group_size() {
            1 => self.layout.shape().into(),
            g => {
                let mut shape = self.layout.shape().to_vec();
                *shape.last_mut().unwrap() *= g;
                shape.into()
            }
        }
    }

    pub fn is_contiguous(&self) -> bool {
        match self.layout.merge_be(0, self.layout.ndim()) {
            Some(layout) => {
                let &[s] = layout.strides() else {
                    unreachable!()
                };
                s == self.dt.nbytes() as isize
            }
            None => false,
        }
    }
}

impl<T, const N: usize> Tensor<T, N> {
    pub fn as_ref(&self) -> Tensor<&T, N> {
        Tensor {
            dt: self.dt,
            layout: self.layout.clone(),
            data: &self.data,
        }
    }

    pub fn as_mut(&mut self) -> Tensor<&mut T, N> {
        Tensor {
            dt: self.dt,
            layout: self.layout.clone(),
            data: &mut self.data,
        }
    }

    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Tensor<U, N> {
        let Self { dt, layout, data } = self;
        Tensor {
            dt,
            layout,
            data: f(data),
        }
    }
}

impl<T: Clone, const N: usize> Tensor<T, N> {
    pub fn cloned(&self) -> Self {
        Tensor {
            dt: self.dt,
            layout: self.layout.clone(),
            data: self.data.clone(),
        }
    }
}

impl<T: Deref, const N: usize> Tensor<T, N> {
    pub fn as_deref(&self) -> Tensor<&<T as Deref>::Target, N> {
        Tensor {
            dt: self.dt,
            layout: self.layout.clone(),
            data: self.data.deref(),
        }
    }
}

impl<T: DerefMut, const N: usize> Tensor<T, N> {
    pub fn as_deref_mut(&mut self) -> Tensor<&mut <T as Deref>::Target, N> {
        Tensor {
            dt: self.dt,
            layout: self.layout.clone(),
            data: self.data.deref_mut(),
        }
    }
}

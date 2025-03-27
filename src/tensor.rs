use digit_layout::DigitLayout;
use ndarray_layout::{ArrayLayout, Endian::BigEndian};
use std::{
    ops::{Deref, DerefMut},
    ptr::copy_nonoverlapping,
    slice::{from_raw_parts, from_raw_parts_mut},
};

pub struct Tensor<T> {
    dt: DigitLayout,
    layout: ArrayLayout<4>,
    data: T,
}

impl Tensor<usize> {
    pub fn new(dt: DigitLayout, shape: &[usize]) -> Self {
        assert_eq!(dt.group_size(), 1);
        let layout = ArrayLayout::new_contiguous(shape, BigEndian, dt.nbytes());
        let size = layout.num_elements() * dt.nbytes();
        Self {
            dt,
            layout,
            data: size,
        }
    }
}

impl<T> Tensor<T> {
    pub const fn dt(&self) -> DigitLayout {
        self.dt
    }

    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    pub fn layout(&self) -> &ArrayLayout<4> {
        &self.layout
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

    pub fn get(&self) -> &T {
        assert!(self.is_contiguous());

        &self.data
    }

    pub fn get_mut(&mut self) -> &mut T {
        assert!(self.is_contiguous());

        &mut self.data
    }

    pub fn take(self) -> T {
        assert!(self.is_contiguous());

        self.data
    }

    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Tensor<U> {
        assert!(self.is_contiguous());

        Tensor {
            dt: self.dt,
            layout: self.layout,
            data: f(self.data),
        }
    }

    pub fn as_ref(&self) -> Tensor<&T> {
        Tensor {
            dt: self.dt,
            layout: self.layout.clone(),
            data: &self.data,
        }
    }

    pub fn index(self, indices: &[usize]) -> Self {
        let mut layout = self.layout.clone();
        for &index in indices {
            layout = layout.index(0, index)
        }
        Self {
            dt: self.dt,
            layout,
            data: self.data,
        }
    }

    pub fn merge(self, axis: usize, len: usize) -> Self {
        Self {
            dt: self.dt,
            layout: self.layout.merge_be(axis, len).unwrap(),
            data: self.data,
        }
    }

    pub fn slice(self, axis: usize, start: usize, len: usize) -> Self {
        Self {
            dt: self.dt,
            layout: self.layout.slice(axis, start, 1, len),
            data: self.data,
        }
    }
}

impl<T: Deref<Target = [u8]>> Tensor<T> {
    pub fn as_slice(&self) -> Tensor<&[u8]> {
        Tensor {
            dt: self.dt,
            layout: self.layout.clone(),
            data: &self.data,
        }
    }
}

impl<T: DerefMut<Target = [u8]>> Tensor<T> {
    pub fn as_slice_mut(&mut self) -> Tensor<&mut [u8]> {
        Tensor {
            dt: self.dt,
            layout: self.layout.clone(),
            data: &mut self.data,
        }
    }
}

impl<'s> Tensor<&'s [u8]> {
    pub fn scalar<U>(&self) -> &'s U {
        let &[] = self.layout.shape() else {
            panic!("not a scalar tensor")
        };

        unsafe { &*self.ptr() }
    }

    pub fn vector<U>(&self) -> &'s [U] {
        let &[n] = self.layout.shape() else {
            panic!("not a vector tensor")
        };
        assert!(self.is_contiguous());

        unsafe { from_raw_parts(self.ptr(), n) }
    }

    pub fn ptr<U>(&self) -> *const U {
        assert_eq!(self.dt.nbytes(), size_of::<U>());
        self.data[self.layout.offset() as usize..].as_ptr().cast()
    }
}

impl<'s> Tensor<&'s mut [u8]> {
    pub fn scalar_mut<U>(&mut self) -> &'s mut U {
        let &[] = self.layout.shape() else {
            panic!("not a scalar tensor")
        };

        unsafe { &mut *self.mut_ptr() }
    }

    pub fn vector_mut<U>(&mut self) -> &'s mut [U] {
        let &[n] = self.layout.shape() else {
            panic!("not a vector tensor: {:?}", self.layout.shape())
        };
        assert!(self.is_contiguous());

        unsafe { from_raw_parts_mut(self.mut_ptr(), n) }
    }

    pub fn write<U: Copy>(&mut self, data: &[U]) {
        assert!(self.is_contiguous());

        let buf = self.data.deref_mut();
        assert_eq!(size_of_val(buf), buf.len());
        unsafe { copy_nonoverlapping(data.as_ptr().cast::<u8>(), buf.as_mut_ptr(), buf.len()) }
    }

    pub fn mut_ptr<U>(&mut self) -> *mut U {
        assert_eq!(self.dt.nbytes(), size_of::<U>());
        self.data[self.layout.offset() as usize..]
            .as_mut_ptr()
            .cast()
    }
}

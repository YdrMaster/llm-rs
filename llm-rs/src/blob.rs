use std::{
    alloc::{Layout, alloc, alloc_zeroed, dealloc},
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

pub struct Blob {
    ptr: NonNull<u8>,
    len: usize,
}

impl Blob {
    pub fn new(len: usize) -> Self {
        Self {
            ptr: NonNull::new(unsafe {
                alloc(Layout::from_size_align(len, align_of::<usize>()).unwrap())
            })
            .unwrap(),
            len,
        }
    }

    pub fn new_zeroed(len: usize) -> Self {
        Self {
            ptr: NonNull::new(unsafe {
                alloc_zeroed(Layout::from_size_align(len, align_of::<usize>()).unwrap())
            })
            .unwrap(),
            len,
        }
    }
}

impl Drop for Blob {
    fn drop(&mut self) {
        unsafe {
            dealloc(
                self.ptr.as_ptr(),
                Layout::from_size_align(self.len, align_of::<usize>()).unwrap(),
            )
        };
    }
}

impl Deref for Blob {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl DerefMut for Blob {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy> From<&[T]> for Blob {
    fn from(value: &[T]) -> Self {
        let len = size_of_val(value);
        let mut blob = Self::new(len);
        unsafe {
            std::ptr::copy_nonoverlapping(value.as_ptr().cast::<u8>(), blob.as_mut_ptr(), len)
        }
        blob
    }
}

impl Clone for Blob {
    fn clone(&self) -> Self {
        let mut ans = Self::new(self.len);
        ans.copy_from_slice(self);
        ans
    }
}

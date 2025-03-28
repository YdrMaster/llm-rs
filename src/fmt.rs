use crate::tensor::Tensor;
use digit_layout::types;
use std::fmt;

impl fmt::Display for Tensor<&[u8]> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let layout = self.layout();
        let ptr = self.get().as_ptr();
        match self.dt() {
            types::F32 => unsafe { layout.write_array(f, ptr.cast::<DataFmt<f32>>()) },
            types::F64 => unsafe { layout.write_array(f, ptr.cast::<DataFmt<f64>>()) },
            types::U32 => unsafe { layout.write_array(f, ptr.cast::<DataFmt<u32>>()) },
            types::U64 => unsafe { layout.write_array(f, ptr.cast::<DataFmt<u64>>()) },
            _ => todo!(),
        }
    }
}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct DataFmt<T: Copy>(T);

impl fmt::Display for DataFmt<f32> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0 == 0. {
            write!(f, " ________")
        } else {
            write!(f, "{:>9.3e}", self.0)
        }
    }
}

impl fmt::Display for DataFmt<f64> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0 == 0. {
            write!(f, " ________")
        } else {
            write!(f, "{:>9.3e}", self.0)
        }
    }
}

impl fmt::Display for DataFmt<u32> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0 == 0 {
            write!(f, " ________")
        } else {
            write!(f, "{:>6}", self.0)
        }
    }
}

impl fmt::Display for DataFmt<u64> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0 == 0 {
            write!(f, " ________")
        } else {
            write!(f, "{:>6}", self.0)
        }
    }
}

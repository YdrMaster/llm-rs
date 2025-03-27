use crate::tensor::Tensor;
use digit_layout::types;
use std::fmt;

pub trait DataFmt {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result;
}

impl DataFmt for f32 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if *self == 0. {
            write!(f, " ________")
        } else {
            write!(f, "{self:>9.3e}")
        }
    }
}

impl DataFmt for f64 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if *self == 0. {
            write!(f, " ________")
        } else {
            write!(f, "{self:>9.3e}")
        }
    }
}

impl DataFmt for u32 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if *self == 0 {
            write!(f, " ________")
        } else {
            write!(f, "{self:>6}")
        }
    }
}

impl DataFmt for u64 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if *self == 0 {
            write!(f, " ________")
        } else {
            write!(f, "{self:>6}")
        }
    }
}

impl fmt::Display for Tensor<&[u8]> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.dt() {
            types::F32 => self.write_tensor::<f32>(&mut vec![], f),
            types::F64 => self.write_tensor::<f64>(&mut vec![], f),
            types::U32 => self.write_tensor::<u32>(&mut vec![], f),
            types::U64 => self.write_tensor::<u64>(&mut vec![], f),
            _ => todo!(),
        }
    }
}

impl Tensor<&[u8]> {
    fn write_tensor<T: DataFmt>(
        &self,
        indices: &mut Vec<[usize; 2]>,
        f: &mut fmt::Formatter,
    ) -> fmt::Result {
        let ptr = self.ptr::<T>();
        match *self.shape() {
            [] => {
                write!(f, "<>")?;
                write_indices(f, indices)?;
                write_matrix(f, ptr, [1, 1], [1, 1])
            }
            [len] => {
                let &[stride] = self.layout().strides() else {
                    unreachable!()
                };
                write!(f, "<{len}>")?;
                write_indices(f, indices)?;
                write_matrix(f, ptr, [len, 1], [stride, 1])
            }
            [rows, cols] => {
                let &[rs, cs] = self.layout().strides() else {
                    unreachable!()
                };
                write!(f, "<{rows}x{cols}>")?;
                write_indices(f, indices)?;
                write_matrix(f, ptr, [rows, cols], [rs, cs])
            }
            [batch, ..] => {
                for i in 0..batch {
                    indices.push([i, batch]);
                    self.as_slice().index(&[i]).write_tensor::<T>(indices, f)?;
                    indices.pop();
                }
                Ok(())
            }
        }
    }
}

fn write_matrix<T: DataFmt>(
    f: &mut fmt::Formatter,
    ptr: *const T,
    shape: [usize; 2],
    strides: [isize; 2],
) -> fmt::Result {
    let [rows, cols] = shape;
    let [rs, cs] = strides;
    for r in 0..rows as isize {
        for c in 0..cols as isize {
            unsafe { &*ptr.byte_offset(r * rs + c * cs) }.fmt(f)?;
            write!(f, " ")?;
        }
        writeln!(f)?;
    }
    Ok(())
}

fn write_indices(f: &mut fmt::Formatter, indices: &[[usize; 2]]) -> fmt::Result {
    for &[i, b] in indices {
        write!(f, ", {i}/{b}")?;
    }
    writeln!(f)
}

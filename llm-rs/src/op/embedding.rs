trait Index: Copy + Sync {
    fn as_usize(self) -> usize;
}

impl<T: Copy + Sync + Into<usize>> Index for T {
    fn as_usize(self) -> usize {
        self.into()
    }
}

pub struct BatchIter {
    batch_size: usize,
    seq_len: usize,
    index: usize,
}

impl BatchIter {
    pub fn new(batch_size: usize, seq_len: usize) -> Self {
        Self {
            batch_size,
            seq_len,
            index: 0,
        }
    }
}

impl Iterator for BatchIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let &mut Self {
            batch_size,
            seq_len,
            index,
        } = self;
        if index / seq_len < batch_size {
            self.index += 1;
            Some(index % seq_len)
        } else {
            None
        }
    }
}

pub fn build_pos(buf: &mut [u8], nseqs: impl IntoIterator<Item = usize>) {
    let ([], mut slice, []) = (unsafe { buf.align_to_mut::<u16>() }) else {
        unreachable!()
    };
    for i in nseqs {
        slice[0] = i as _;
        slice = &mut slice[1..]
    }
}

pub mod forward {
    use super::Index;
    use crate::op::Tensor;
    use crate::op::{macros::*, unique};
    use digit_layout::types;
    use std::{iter::zip, ops::Add};

    pub(crate) fn embedding(
        y: &Tensor,
        i1: &Tensor,
        i2: &Tensor,
        table1: &Tensor,
        table2: &Tensor,
    ) {
        clone_tensor!(y i1 i2 table1 table2);

        dims!([n0, d0] = y);
        dims!([n1] = i1);
        dims!([n2] = i2);
        dims!([_nt1, d1] = table1);
        dims!([_nt2, d2] = table2);

        let n = unique(&[n0, n1, n2]).unwrap();
        let d = unique(&[d0, d1, d2]).unwrap();

        strides!([nsy, dsy] = y);
        strides!([ns1] = i1);
        strides!([ns2] = i2);

        assert_eq!(dsy, y.dt().nbytes() as isize);
        assert_eq!(ns1, i1.dt().nbytes() as isize);
        assert_eq!(ns2, i2.dt().nbytes() as isize);
        assert!(table1.is_contiguous());
        assert!(table2.is_contiguous());

        let scheme = Scheme {
            n,
            d,
            nsy,
            y: y.as_ref().map(|b| &mut **b.write()).mut_ptr(),
            i1: i1.as_ref().map(|b| &**b.read()).ptr(),
            i2: i2.as_ref().map(|b| &**b.read()).ptr(),
            table1: table1.as_ref().map(|b| &**b.read()).ptr(),
            table2: table2.as_ref().map(|b| &**b.read()).ptr(),
        };

        match (y.dt(), i1.dt(), i2.dt()) {
            (types::F32, types::U16, types::U16) => scheme.compute::<f32, u16, u16>(),
            (_, _, _) => todo!(),
        }
    }

    struct Scheme {
        n: usize,
        d: usize,
        nsy: isize,
        y: *mut u8,
        i1: *const u8,
        i2: *const u8,
        table1: *const u8,
        table2: *const u8,
    }

    impl Scheme {
        fn compute<T: Add<Output = T>, I1: Index, I2: Index>(&self) {
            let &Self {
                n,
                d,
                nsy,
                y,
                i1,
                i2,
                table1,
                table2,
            } = self;
            let i1 = unsafe { std::slice::from_raw_parts(i1.cast::<I1>(), n) };
            let i2 = unsafe { std::slice::from_raw_parts(i2.cast::<I2>(), n) };
            for (i, (i1, i2)) in zip(i1, i2).enumerate() {
                let y = unsafe { y.byte_offset(nsy * i as isize) }.cast::<T>();
                let x1 = unsafe { table1.byte_add(i1.as_usize() * d * size_of::<T>()) }.cast::<T>();
                let x2 = unsafe { table2.byte_add(i2.as_usize() * d * size_of::<T>()) }.cast::<T>();
                for i in 0..d {
                    unsafe { y.add(i).write(x1.add(i).read() + x2.add(i).read()) }
                }
            }
        }
    }
}

pub mod backward {
    use super::Index;
    use crate::op::Tensor;
    use crate::op::{macros::*, unique};
    use digit_layout::types;
    use std::{iter::zip, ops::AddAssign};

    pub(crate) fn embedding(
        dtable1: &Tensor,
        dtable2: &Tensor,
        dy: &Tensor,
        i1: &Tensor,
        i2: &Tensor,
    ) {
        clone_tensor!(dtable1 dtable2 dy i1 i2);

        dims!([_nt1, d1] = dtable1);
        dims!([_nt2, d2] = dtable2);
        dims!([n0, d0] = dy);
        dims!([n1] = i1);
        dims!([n2] = i2);

        let n = unique(&[n0, n1, n2]).unwrap();
        let d = unique(&[d0, d1, d2]).unwrap();

        strides!([nsy, dsy] = dy);
        strides!([ns1] = i1);
        strides!([ns2] = i2);

        assert!(dtable1.is_contiguous());
        assert!(dtable2.is_contiguous());
        assert_eq!(dsy, dy.dt().nbytes() as isize);
        assert_eq!(ns1, i1.dt().nbytes() as isize);
        assert_eq!(ns2, i2.dt().nbytes() as isize);

        let scheme = Scheme {
            n,
            d,
            nsy,
            dtable1: dtable1.as_ref().map(|b| &mut **b.write()).mut_ptr(),
            dtable2: dtable2.as_ref().map(|b| &mut **b.write()).mut_ptr(),
            dy: dy.as_ref().map(|b| &**b.read()).ptr(),
            i1: i1.as_ref().map(|b| &**b.read()).ptr(),
            i2: i2.as_ref().map(|b| &**b.read()).ptr(),
        };

        match (dy.dt(), i1.dt(), i2.dt()) {
            (types::F32, types::U16, types::U16) => scheme.compute::<f32, u16, u16>(),
            (_, _, _) => todo!(),
        }
    }

    struct Scheme {
        n: usize,
        d: usize,
        nsy: isize,
        dtable1: *mut u8,
        dtable2: *mut u8,
        dy: *const u8,
        i1: *const u8,
        i2: *const u8,
    }

    impl Scheme {
        fn compute<T: AddAssign + Copy, I1: Index, I2: Index>(&self) {
            let &Self {
                n,
                d,
                nsy,
                dtable1,
                dtable2,
                dy,
                i1,
                i2,
            } = self;
            let i1 = unsafe { std::slice::from_raw_parts(i1.cast::<I1>(), n) };
            let i2 = unsafe { std::slice::from_raw_parts(i2.cast::<I2>(), n) };
            for (i, (i1, i2)) in zip(i1, i2).enumerate() {
                let dy = unsafe { dy.byte_offset(nsy * i as isize) }.cast::<T>();
                let x1 =
                    unsafe { dtable1.byte_add(i1.as_usize() * d * size_of::<T>()) }.cast::<T>();
                let x2 =
                    unsafe { dtable2.byte_add(i2.as_usize() * d * size_of::<T>()) }.cast::<T>();
                for i in 0..d {
                    unsafe { *x1.add(i) += *dy.add(i) }
                    unsafe { *x2.add(i) += *dy.add(i) }
                }
            }
        }
    }
}

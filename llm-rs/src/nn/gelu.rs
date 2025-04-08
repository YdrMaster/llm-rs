use super::{NeuralNetwork, macros::*, unique};
use crate::{Blob, Context, Tensor};
use digit_layout::types;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{
    f32::consts::PI,
    ops::{AddAssign, Mul},
    sync::LazyLock,
};
use tensor::rw_rc::RwRc;

pub struct Gelu {
    x: Option<RwRc<Tensor<Blob>>>,
}

impl NeuralNetwork for Gelu {
    type Init = ();

    fn init(_init: Self::Init, _ctx: &mut Context) -> Self {
        Self { x: None }
    }

    fn forward(
        &mut self,
        inputs: impl IntoIterator<Item = RwRc<Tensor<Blob>>>,
        ctx: &mut Context,
    ) -> Vec<RwRc<Tensor<Blob>>> {
        destruct!([x] = inputs);
        self.x.replace(x);
        let Self { x } = self;

        let x = x.as_ref().unwrap().read();
        let mut y = Tensor::contiguous_of(x).map(Blob::new);

        {
            let y = y.as_deref_mut().merge(0, 2);
            let x = x.as_deref().merge(0, 2);
            ctx.bench(|| forward::gelu(y, x))
        }

        vec![y.share()]
    }

    fn backward(
        &mut self,
        inputs: impl IntoIterator<Item = RwRc<Tensor<Blob>>>,
        ctx: &mut Context,
    ) -> Vec<RwRc<Tensor<Blob>>> {
        destruct!([dy] = inputs);
        let Self { x } = self;

        let x = x.take().unwrap();
        let x = x.read();
        let mut dx = Tensor::contiguous_of(x).map(Blob::new_zeroed);

        {
            let dx = dx.as_deref_mut().merge(0, 2);
            let x = x.as_deref().merge(0, 2);
            let dy = dy.read().as_deref().merge(0, 2);
            ctx.bench(|| backward::gelu(dx, x, dy))
        }

        vec![dx.share()]
    }
}

const GELU_MAGIC: f32 = 0.044715;
static GELU_FACTOR: LazyLock<f32> = LazyLock::new(|| (2. / PI).sqrt());

trait GeluData: Copy + AddAssign + Mul<Output = Self> {
    fn compute(self) -> Self;
    fn grad(self) -> Self;
}

impl GeluData for f32 {
    fn compute(self) -> Self {
        let x3 = GELU_MAGIC * self.powi(3);
        let tanh = *GELU_FACTOR * (self + x3);

        0.5 * self * (1. + tanh.tanh())
    }

    fn grad(self) -> Self {
        let x3 = GELU_MAGIC * self.powi(3);
        let tanh = *GELU_FACTOR * (self + x3);

        let dx3 = 3. * GELU_MAGIC * self.powi(2);
        let dtanh = *GELU_FACTOR * (1. + dx3);

        0.5 * (1. + tanh.tanh()) + 0.5 * self * tanh.cosh().powi(-2) * dtanh
    }
}

mod forward {
    use super::*;

    pub(super) fn gelu(mut y: Tensor<&mut [u8]>, x: Tensor<&[u8]>) {
        dims!([n, d] = y);
        dims!([n_, d_] = x);

        assert_eq!(n, n_);
        assert_eq!(d, d_);

        strides!([nsy, dsy] = y);
        strides!([nsx, dsx] = y);

        let dt = unique(&[y.dt(), x.dt()]).unwrap();

        let scheme = Scheme {
            n,
            d,
            sy: [nsy, dsy],
            sx: [nsx, dsx],
            y: y.mut_ptr(),
            x: x.ptr(),
        };

        match dt {
            types::F32 => scheme.compute::<f32>(),
            _ => todo!(),
        }
    }

    struct Scheme {
        n: usize,
        d: usize,
        sy: [isize; 2],
        sx: [isize; 2],
        y: *mut u8,
        x: *const u8,
    }

    impl Scheme {
        fn compute<T: GeluData>(&self) {
            let &Self { n, d, sy, sx, y, x } = self;
            let y = y as usize;
            let x = x as usize;
            (0..n * d).into_par_iter().for_each(|i| {
                let j = (i % d) as isize;
                let i = (i / d) as isize;
                let [si, sj] = sy;
                let y = unsafe { (y as *mut T).byte_offset(i * si + j * sj) };
                let [si, sj] = sx;
                let x = unsafe { (x as *const T).byte_offset(i * si + j * sj) };
                unsafe { *y = (*x).compute() }
            });
        }
    }
}

mod backward {
    use super::*;

    pub(super) fn gelu(mut dx: Tensor<&mut [u8]>, x: Tensor<&[u8]>, dy: Tensor<&[u8]>) {
        dims!([n0, d0] = dx);
        dims!([n1, d1] = x);
        dims!([n2, d2] = dy);

        let dt = unique(&[dx.dt(), x.dt(), dy.dt()]).unwrap();
        let n = unique(&[n0, n1, n2]).unwrap();
        let d = unique(&[d0, d1, d2]).unwrap();

        strides!([nsdx, dsdx] = dx);
        strides!([nsx, dsx] = x);
        strides!([nsdy, dsdy] = dy);

        let scheme = Scheme {
            n,
            d,
            sdx: [nsdx, dsdx],
            sx: [nsx, dsx],
            sdy: [nsdy, dsdy],
            dx: dx.mut_ptr(),
            x: x.ptr(),
            dy: dy.ptr(),
        };

        match dt {
            types::F32 => scheme.compute::<f32>(),
            _ => todo!(),
        }
    }

    struct Scheme {
        n: usize,
        d: usize,
        sdx: [isize; 2],
        sx: [isize; 2],
        sdy: [isize; 2],
        dx: *mut u8,
        x: *const u8,
        dy: *const u8,
    }

    impl Scheme {
        fn compute<T: GeluData>(&self) {
            let &Self {
                n,
                d,
                sdx,
                sx,
                sdy,
                dx,
                x,
                dy,
            } = self;
            let dx = dx as usize;
            let x = x as usize;
            let dy = dy as usize;
            (0..n * d).into_par_iter().for_each(|i| {
                let j = (i % d) as isize;
                let i = (i / d) as isize;
                let [si, sj] = sdx;
                let dx = unsafe { (dx as *mut T).byte_offset(i * si + j * sj) };
                let [si, sj] = sx;
                let x = unsafe { (x as *const T).byte_offset(i * si + j * sj) };
                let [si, sj] = sdy;
                let dy = unsafe { (dy as *const T).byte_offset(i * si + j * sj) };
                unsafe { *dx += *dy * (*x).grad() }
            });
        }
    }
}

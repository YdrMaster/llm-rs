use super::{NeuralNetwork, macros::*, unique};
use crate::{Blob, Context, Tensor};
use digit_layout::types;
use itertools::izip;
use std::{f32::consts::PI, iter::zip, sync::LazyLock};
use tensor::rw_rc::RwRc;

pub struct Gelu {
    x: Option<RwRc<Tensor<Blob>>>,
}

const GELU_MAGIC: f32 = 0.044715;
static GELU_FACTOR: LazyLock<f32> = LazyLock::new(|| (2. / PI).sqrt());

impl NeuralNetwork for Gelu {
    type Init = ();

    fn init(_init: Self::Init, _ctx: &mut Context) -> Self {
        Self { x: None }
    }

    fn forward(
        &mut self,
        inputs: impl IntoIterator<Item = RwRc<Tensor<Blob>>>,
        _ctx: &mut Context,
    ) -> Vec<RwRc<Tensor<Blob>>> {
        destruct!([x] = inputs);
        self.x.replace(x);
        let Self { x } = self;

        let x = x.as_ref().unwrap().read();
        let mut y = Tensor::contiguous_of(x).map(Blob::new);

        forward(y.as_deref_mut(), x.as_deref());

        vec![y.share()]
    }

    fn backward(
        &mut self,
        inputs: impl IntoIterator<Item = RwRc<Tensor<Blob>>>,
        _ctx: &mut Context,
    ) -> Vec<RwRc<Tensor<Blob>>> {
        destruct!([dy] = inputs);
        let Self { x } = self;

        let x = x.take().unwrap();
        let x = x.read();
        let mut dx = Tensor::contiguous_of(x).map(Blob::new);

        backward(dx.as_deref_mut(), x.as_deref(), dy.read().as_deref());

        vec![dx.share()]
    }
}

fn forward(y: Tensor<&mut [u8]>, x: Tensor<&[u8]>) {
    let dt = unique(&[y.dt(), x.dt()]).unwrap();
    assert_eq!(dt, types::F32);

    let y = y.merge(0, 3).vector_mut::<f32>();
    let x = x.merge(0, 3).vector::<f32>();
    for (y, x) in zip(y, x) {
        *y = x * 0.5 * (1. + (*GELU_FACTOR * (x + GELU_MAGIC * x.powi(3))).tanh())
    }
}

fn backward(dx: Tensor<&mut [u8]>, x: Tensor<&[u8]>, dy: Tensor<&[u8]>) {
    let dt = unique(&[dx.dt(), x.dt(), dy.dt()]).unwrap();
    assert_eq!(dt, types::F32);

    let dx = dx.merge(0, 3).vector_mut::<f32>();
    let x = x.merge(0, 3).vector::<f32>();
    let dy = dy.merge(0, 3).vector::<f32>();

    for (dx, x, dy) in izip!(dx, x, dy) {
        let cube = GELU_MAGIC * x.powi(3);
        let tanh_arg = *GELU_FACTOR * (x + cube);
        let tanh_val = tanh_arg.tanh();
        let cosh_val = tanh_arg.cosh();
        let sech_val = 1. / cosh_val.powi(2);
        let grad = 0.5 * (1. + tanh_val)
            + 0.5 * x * sech_val * *GELU_FACTOR * (1. + 3. * GELU_MAGIC * x.powi(2));
        *dx += *dy * grad
    }
}

use crate::{Tensor, blob::Blob};
use digit_layout::types;
use itertools::izip;
use std::collections::HashMap;
use tensor::rw_rc::{RwRc, RwWeak};

pub trait Optimizer {
    fn update(&mut self, weight: RwRc<Tensor<Blob>>, gradient: RwRc<Tensor<Blob>>);
}

pub struct AdamW {
    weights: HashMap<WeakWeight, State>,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    t: i32,
}

type WeakWeight = RwWeak<Tensor<Blob>>;

struct State {
    m: Blob,
    v: Blob,
}

impl Optimizer for AdamW {
    fn update(&mut self, weight: RwRc<Tensor<Blob>>, gradient: RwRc<Tensor<Blob>>) {
        let &mut Self {
            ref mut weights,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            t,
        } = self;
        let State { m, v } = weights.entry(weight.weak()).or_insert_with(|| {
            let len = Tensor::contiguous_of(weight.read()).take();
            State {
                m: Blob::new_zeroed(len),
                v: Blob::new_zeroed(len),
            }
        });

        let weight = weight.write();
        let gradient = gradient.read();

        assert_eq!(weight.dt(), types::F32);
        assert_eq!(gradient.dt(), types::F32);
        assert_eq!(weight.shape(), gradient.shape());

        let ndim = weight.layout().ndim();
        let weight = weight.as_deref_mut().merge(0, ndim).vector_mut::<f32>();
        let gradient = gradient.as_deref().merge(0, ndim).vector::<f32>();
        let ([], m, []) = (unsafe { m.align_to_mut::<f32>() }) else {
            unreachable!()
        };
        let ([], v, []) = (unsafe { v.align_to_mut::<f32>() }) else {
            unreachable!()
        };

        let hat1 = 1. / (1. - beta1.powi(t));
        let hat2 = 1. / (1. - beta2.powi(t));
        for (w, g, m, v) in izip!(weight, gradient, m, v) {
            *m = beta1 * *m + (1. - beta1) * g;
            *v = beta2 * *v + (1. - beta2) * g * g;
            *w -= learning_rate * (*m * hat1 / ((*v * hat2).sqrt() + epsilon) + weight_decay * *w)
        }
    }
}

impl AdamW {
    pub fn new(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> Self {
        Self {
            weights: Default::default(),
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            t: 1,
        }
    }

    pub fn next(&mut self) {
        self.t += 1
    }
}

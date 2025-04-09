use crate::{Blob, HashWeak, Tensor, nn::NeuralNetwork, optimizer::Optimizer};
use digit_layout::DigitLayout;
use rw_rc::RwRc;
use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
    time::Instant,
};

pub struct Context {
    path: String,
    weights: HashMap<HashWeak<Tensor<RwRc<Blob>>>, WeightInfo>,
    bench: bool,
}

#[derive(Default)]
struct WeightInfo {
    gradient: Option<Rc<Tensor<RwRc<Blob>>>>,
    names: HashSet<String>,
}

impl Context {
    pub fn new(bench: bool) -> Self {
        Self {
            path: "Ω".into(),
            weights: Default::default(),
            bench,
        }
    }

    pub fn trap<T>(&mut self, sub: impl AsRef<str>, f: impl FnOnce(&mut Self) -> T) -> T {
        let sub = sub.as_ref();

        self.path.push('.');
        self.path.push_str(sub);

        let ans = f(self);

        assert!(self.path.ends_with(sub));
        self.path.truncate(self.path.len() - sub.len() - 1);

        ans
    }

    pub fn write_gradient(
        &mut self,
        name: &str,
        weight: &Rc<Tensor<RwRc<Blob>>>,
    ) -> Rc<Tensor<RwRc<Blob>>> {
        // 注册权重
        let info = self
            .weights
            .entry(HashWeak(Rc::downgrade(weight)))
            .or_default();
        // 记录名字
        info.names.insert(format!("{}:{name}", self.path));
        // 生成或取出梯度
        info.gradient
            .get_or_insert_with(|| {
                Tensor::contiguous_of(weight)
                    .map(Blob::new_zeroed)
                    .map(RwRc::new)
                    .share()
            })
            .clone()
    }

    pub fn zero_grad(&mut self) {
        for info in self.weights.values_mut() {
            let _ = info.gradient.take();
        }
    }

    pub fn update(&self, optimizer: &mut impl Optimizer) {
        for (weak, info) in &self.weights {
            let weight = weak.0.upgrade().unwrap();
            let gradient = info.gradient.clone().unwrap();
            optimizer.update(weight, gradient)
        }
    }
}

impl Context {
    pub fn init<NN: NeuralNetwork>(&mut self, name: impl AsRef<str>, init: NN::Init) -> NN {
        self.trap(name, |ctx| NN::init(init, ctx))
    }

    pub fn forward<NN: NeuralNetwork>(
        &mut self,
        name: impl AsRef<str>,
        nn: &mut NN,
        inputs: impl IntoIterator<Item = Rc<Tensor<RwRc<Blob>>>>,
    ) -> Vec<Rc<Tensor<RwRc<Blob>>>> {
        self.trap(name, |ctx| nn.forward(inputs, ctx))
    }

    pub fn backward<NN: NeuralNetwork>(
        &mut self,
        name: impl AsRef<str>,
        nn: &mut NN,
        inputs: impl IntoIterator<Item = Rc<Tensor<RwRc<Blob>>>>,
    ) -> Vec<Rc<Tensor<RwRc<Blob>>>> {
        self.trap(name, |ctx| nn.backward(inputs, ctx))
    }

    pub fn tensor(&self, dt: DigitLayout, shape: &[usize]) -> Tensor<RwRc<Blob>> {
        Tensor::new(dt, shape).map(Blob::new).map(RwRc::new)
    }

    pub fn tensor_zeroed(&self, dt: DigitLayout, shape: &[usize]) -> Tensor<RwRc<Blob>> {
        Tensor::new(dt, shape).map(Blob::new_zeroed).map(RwRc::new)
    }

    pub fn bench(&self, f: impl FnOnce()) {
        let time = Instant::now();
        f();
        if self.bench {
            println!("{}: {:?}", self.path, time.elapsed())
        }
    }
}

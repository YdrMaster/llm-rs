use crate::{
    HashWeak,
    nn::NeuralNetwork,
    optimizer::Optimizer,
    vm::{Tensor, TestVM, VirtualMachine},
};
use digit_layout::DigitLayout;
use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
    time::Instant,
};

pub type TestContext = Context<TestVM>;

pub struct Context<VM: VirtualMachine> {
    vm: VM,
    path: String,
    weights: HashMap<HashWeak<VM::Tensor>, WeightInfo<VM::Tensor>>,
    bench: bool,
}

struct WeightInfo<T> {
    gradient: Option<Rc<T>>,
    names: HashSet<String>,
}

impl<T> Default for WeightInfo<T> {
    fn default() -> Self {
        Self {
            gradient: Default::default(),
            names: Default::default(),
        }
    }
}

impl<VM: VirtualMachine> Context<VM> {
    pub fn new(vm: VM, bench: bool) -> Self {
        Self {
            vm,
            path: "Ω".into(),
            weights: Default::default(),
            bench,
        }
    }

    pub fn write_gradient(&mut self, name: &str, weight: &Rc<VM::Tensor>) -> Rc<VM::Tensor> {
        // 注册权重
        let info = self
            .weights
            .entry(HashWeak(Rc::downgrade(weight)))
            .or_default();
        // 记录名字
        info.names.insert(format!("{}:{name}", self.path));
        // 生成或取出梯度
        info.gradient
            .get_or_insert_with(|| Rc::new(self.vm.tensor_zeroed(weight.dt(), &weight.shape())))
            .clone()
    }

    pub fn zero_grad(&mut self) {
        for info in self.weights.values_mut() {
            let _ = info.gradient.take();
        }
    }

    pub fn update(&self, optimizer: &mut impl Optimizer<VM::Tensor>) {
        for (weak, info) in &self.weights {
            let weight = weak.0.upgrade().unwrap();
            let gradient = info.gradient.clone().unwrap();
            optimizer.update(weight, gradient)
        }
    }
}

impl<VM: VirtualMachine> Context<VM> {
    pub fn init<NN: NeuralNetwork<VM>>(&mut self, name: impl AsRef<str>, init: NN::Init) -> NN {
        self.trap(name, |ctx| NN::init(init, ctx))
    }

    pub fn forward<NN: NeuralNetwork<VM>>(
        &mut self,
        name: impl AsRef<str>,
        nn: &mut NN,
        inputs: impl IntoIterator<Item = Rc<VM::Tensor>>,
    ) -> Vec<Rc<VM::Tensor>> {
        self.trap(name, |ctx| nn.forward(inputs, ctx))
    }

    pub fn backward<NN: NeuralNetwork<VM>>(
        &mut self,
        name: impl AsRef<str>,
        nn: &mut NN,
        inputs: impl IntoIterator<Item = Rc<VM::Tensor>>,
    ) -> Vec<Rc<VM::Tensor>> {
        self.trap(name, |ctx| nn.backward(inputs, ctx))
    }

    pub fn tensor(&self, dt: DigitLayout, shape: &[usize]) -> VM::Tensor {
        self.vm.tensor(dt, shape)
    }

    pub fn tensor_zeroed(&self, dt: DigitLayout, shape: &[usize]) -> VM::Tensor {
        self.vm.tensor_zeroed(dt, shape)
    }

    pub fn bench(&self, f: impl FnOnce()) {
        let time = Instant::now();
        f();
        if self.bench {
            println!("{}: {:?}", self.path, time.elapsed())
        }
    }

    fn trap<T>(&mut self, sub: impl AsRef<str>, f: impl FnOnce(&mut Self) -> T) -> T {
        let sub = sub.as_ref();

        self.path.push('.');
        self.path.push_str(sub);

        let ans = f(self);

        assert!(self.path.ends_with(sub));
        self.path.truncate(self.path.len() - sub.len() - 1);

        ans
    }
}

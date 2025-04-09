use super::Tensor;
use crate::op::macros::clone_tensor;

pub fn add(y: &Tensor, x: &Tensor) {
    clone_tensor!(y x);

    assert_eq!(y.shape(), x.shape());
    let ndim = y.layout().ndim();
    let y = y.as_ref().merge(0, ndim);
    let x = x.as_ref().merge(0, ndim);
    for (y, x) in std::iter::zip(
        y.map(|t| &mut **t.write()).vector_mut::<f32>(),
        x.map(|t| &**t.write()).vector::<f32>(),
    ) {
        *y += x
    }
}

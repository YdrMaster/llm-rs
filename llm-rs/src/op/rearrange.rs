use crate::{nn::Tensor_, op::unique};
use mem_rearrange::Rearranging;

pub fn rearrange(y: &Tensor_, x: &Tensor_) {
    let dt = unique(&[y.dt(), x.dt()]).unwrap();
    assert_eq!(y.shape(), x.shape());

    unsafe {
        Rearranging::new(y.layout(), x.layout(), dt.nbytes())
            .unwrap()
            .launch(y.get().write().as_mut_ptr(), x.get().read().as_ptr())
    }
}

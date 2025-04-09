pub mod add;
pub mod attention;
pub mod embedding;
pub mod gelu;
pub mod gemm;
pub mod layer_norm;
pub mod linear;
pub mod loss;

type Tensor = crate::Tensor<rw_rc::RwRc<crate::Blob>>;

fn unique<T: Copy + Eq>(vals: &[T]) -> Option<T> {
    let [val, tail @ ..] = vals else {
        return None;
    };
    for v in tail {
        if v != val {
            return None;
        }
    }
    Some(*val)
}

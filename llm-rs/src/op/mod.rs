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

mod macros {
    macro_rules! dims {
        ($pat:pat = $tensor:expr) => {
            let &$pat = &*$tensor.shape() else {
                panic!("Ndim mismatch ( = {})", $tensor.shape().len())
            };
        };
    }

    macro_rules! strides {
        ($pat:pat = $tensor:expr) => {
            let &$pat = &*$tensor.layout().strides() else {
                panic!("Ndim mismatch ( = {})", $tensor.layout().strides().len())
            };
        };
    }

    macro_rules! destruct {
        ([$( $name:ident ),+] = $iter:expr) => {
            let mut iter = $iter.into_iter();
            $( let $name = iter.next().unwrap(); )+
            assert!(iter.next().is_none());
        };
    }

    macro_rules! clone_tensor {
        ($( $tensor:ident )+) => {
            $( let $tensor = $tensor.cloned(); )+
        };
    }

    pub(super) use {clone_tensor, destruct, dims, strides};
}

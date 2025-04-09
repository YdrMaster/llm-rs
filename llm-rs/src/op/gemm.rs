use crate::{macros::*, nn::Tensor_, op::unique};
use digit_layout::types;

pub fn mat_mul(c: &Tensor_, beta: f32, a: &Tensor_, b: &Tensor_, alpha: Option<f32>) {
    clone_tensor!(c a b);

    let dt = unique(&[c.dt(), a.dt(), b.dt()]).unwrap();
    assert_eq!(dt, types::F32);

    dims!([m, n] = c);
    dims!([m_, k] = a);
    dims!([k_, n_] = b);
    strides!([rs_c, cs_c] = c);
    strides!([rs_a, cs_a] = a);
    strides!([rs_b, cs_b] = b);

    assert_eq!(m, m_);
    assert_eq!(n, n_);
    assert_eq!(k, k_);

    unsafe {
        gemm::gemm::<f32>(
            m,
            n,
            k,
            c.as_ref().map(|b| &mut **b.write()).mut_ptr(),
            cs_c / dt.nbytes() as isize,
            rs_c / dt.nbytes() as isize,
            alpha.is_some(),
            a.as_ref().map(|b| &**b.read()).ptr(),
            cs_a / dt.nbytes() as isize,
            rs_a / dt.nbytes() as isize,
            b.as_ref().map(|b| &**b.read()).ptr(),
            cs_b / dt.nbytes() as isize,
            rs_b / dt.nbytes() as isize,
            alpha.unwrap_or(0.0),
            beta,
            false,
            false,
            false,
            gemm::Parallelism::Rayon(0),
        )
    }
}

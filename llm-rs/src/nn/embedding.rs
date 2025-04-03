use super::{NeuralNetwork, macros::*, unique};
use crate::{Blob, Context, Tensor};
use digit_layout::types;
use itertools::izip;
use tensor::rw_rc::RwRc;

pub struct Embedding {
    te: RwRc<Tensor<Blob>>,
    pe: RwRc<Tensor<Blob>>,
    tokens: Option<RwRc<Tensor<Blob>>>,
}

impl NeuralNetwork for Embedding {
    type Init = [RwRc<Tensor<Blob>>; 2];

    fn init(init: Self::Init, _ctx: &mut Context) -> Self {
        let [te, pe] = init;
        Self {
            te,
            pe,
            tokens: None,
        }
    }

    fn forward(
        &mut self,
        inputs: impl IntoIterator<Item = RwRc<Tensor<Blob>>>,
        _ctx: &mut Context,
    ) -> Vec<RwRc<Tensor<Blob>>> {
        destruct!([tokens] = inputs);
        self.tokens.replace(tokens);
        let Self { te, pe, tokens } = self;
        let te = te.read();
        let pe = pe.read();
        let tokens = tokens.as_ref().unwrap().read();

        dims!([batch_size, n_seq] = tokens);

        dims!([_, d] = te);
        let mut y = Tensor::new(te.dt(), &[batch_size, n_seq, d]).map(Blob::new);

        forward(
            y.as_deref_mut(),
            tokens.as_deref(),
            te.as_deref(),
            pe.as_deref(),
        );

        vec![y.share()]
    }

    fn backward(
        &mut self,
        inputs: impl IntoIterator<Item = RwRc<Tensor<Blob>>>,
        ctx: &mut Context,
    ) -> Vec<RwRc<Tensor<Blob>>> {
        destruct!([dy] = inputs);
        let Self { te, pe, tokens } = self;

        backward(
            ctx.write_gradient("wte", te).write().as_deref_mut(),
            ctx.write_gradient("wpe", pe).write().as_deref_mut(),
            dy.read().as_deref(),
            tokens.take().unwrap().read().as_deref(),
        );

        te.release();
        pe.release();

        vec![]
    }
}

fn forward(mut y: Tensor<&mut [u8]>, tokens: Tensor<&[u8]>, te: Tensor<&[u8]>, pe: Tensor<&[u8]>) {
    let dt = unique(&[y.dt(), te.dt(), pe.dt()]).unwrap();
    assert_eq!(dt, types::F32);
    assert_eq!(tokens.dt(), types::U16);

    dims!([batch_size, n_seq, d] = y);
    dims!([batch_size_, n_seq_] = tokens);
    dims!([_, d_] = te);
    dims!([_, d__] = pe);

    assert_eq!(batch_size, batch_size_);
    assert_eq!(n_seq, n_seq_);
    unique(&[d, d_, d__]).unwrap();

    for b in 0..batch_size {
        for t in 0..n_seq {
            let out = y.as_deref_mut().index(&[b, t]).vector_mut::<f32>();
            let &ix = tokens.as_deref().index(&[b, t]).scalar::<u16>();
            let wte = te.as_deref().index(&[ix as _]).vector::<f32>();
            let wpe = pe.as_deref().index(&[t]).vector::<f32>();
            for (out, wte, wpe) in izip!(&mut *out, wte, wpe) {
                *out = wte + wpe;
            }
        }
    }
}

fn backward(
    mut dte: Tensor<&mut [u8]>,
    mut dpe: Tensor<&mut [u8]>,
    dy: Tensor<&[u8]>,
    tokens: Tensor<&[u8]>,
) {
    dims!([batch_size, n_seq, d] = dy);
    dims!([batch_size_, n_seq_] = tokens);
    dims!([_, d_] = dte);
    dims!([_, d__] = dpe);
    assert_eq!(batch_size, batch_size_);
    assert_eq!(n_seq, n_seq_);
    unique(&[d, d_, d__]).unwrap();

    for b in 0..batch_size {
        for t in 0..n_seq {
            let dy = dy.as_deref().index(&[b, t]).vector::<f32>();
            let ix = *tokens.as_deref().index(&[b, t]).scalar::<u16>() as usize;
            let dwte = dte.as_deref_mut().index(&[ix]).vector_mut::<f32>();
            let dwpe = dpe.as_deref_mut().index(&[t]).vector_mut::<f32>();
            for (dwte, dwpe, dy) in izip!(dwte, dwpe, dy) {
                *dwte += dy;
                *dwpe += dy;
            }
        }
    }
}

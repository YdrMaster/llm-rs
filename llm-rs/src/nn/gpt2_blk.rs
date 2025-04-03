use super::{
    NeuralNetwork, attention::Attention, gelu::Gelu, layer_norm::LayerNorm, linear::Linear,
    macros::destruct,
};
use crate::{Blob, Context, Tensor, llmc};
use tensor::rw_rc::RwRc;

const ATTN_NORM: &str = "attn_norm";
const ATTN_QKV: &str = "attn_qkv";
const ATTN: &str = "attn";
const ATTN_O: &str = "attn_o";
const FFN_NORM: &str = "ffn_norm";
const FFN_UP: &str = "ffn_up";
const FFN_ACT: &str = "ffn_act";
const FFN_DOWN: &str = "ffn_down";

pub struct Gpt2Blk {
    attn_norm: LayerNorm,
    attn_qkv: Linear,
    attn: Attention,
    attn_o: Linear,
    ffn_norm: LayerNorm,
    ffn_up: Linear,
    ffn_act: Gelu,
    ffn_down: Linear,
}

impl NeuralNetwork for Gpt2Blk {
    type Init = (llmc::Gpt2Blk<Blob>, usize);

    fn init(init: Self::Init, ctx: &mut Context) -> Self {
        let llmc::Gpt2Blk {
            attn_norm,
            attn_qkv,
            ffn_norm,
            attn_o,
            ffn_up,
            ffn_down,
        } = init.0;

        fn share(arr: [Tensor<Blob>; 2]) -> [RwRc<Tensor<Blob>>; 2] {
            arr.map(|t| t.share())
        }

        let attn_norm = ctx.init(ATTN_NORM, share(attn_norm));

        let [w, b] = share(attn_qkv);
        let attn_qkv = ctx.init(ATTN_QKV, (w, Some(b)));

        let attn = ctx.init(ATTN, init.1);

        let [w, b] = share(attn_o);
        let attn_o = ctx.init(ATTN_O, (w, Some(b)));

        let ffn_norm = ctx.init(FFN_NORM, share(ffn_norm));

        let [w, b] = share(ffn_up);
        let ffn_up = ctx.init(FFN_UP, (w, Some(b)));

        let ffn_act = ctx.init(FFN_ACT, ());

        let [w, b] = share(ffn_down);
        let ffn_down = ctx.init(FFN_DOWN, (w, Some(b)));

        Self {
            attn_norm,
            attn_qkv,
            attn,
            attn_o,
            ffn_norm,
            ffn_up,
            ffn_act,
            ffn_down,
        }
    }

    fn forward(
        &mut self,
        inputs: impl IntoIterator<Item = RwRc<Tensor<Blob>>>,
        ctx: &mut Context,
    ) -> Vec<RwRc<Tensor<Blob>>> {
        let Self {
            attn_norm,
            attn_qkv,
            attn,
            attn_o,
            ffn_norm,
            ffn_up,
            ffn_act,
            ffn_down,
        } = self;

        destruct!([residual] = inputs);

        let x = [residual.clone()];
        let x = ctx.forward(ATTN_NORM, attn_norm, x);
        let x = ctx.forward(ATTN_QKV, attn_qkv, x);
        let x = ctx.forward(ATTN, attn, x);
        let x = ctx.forward(ATTN_O, attn_o, x);

        destruct!([x] = x);
        add(x.write().as_deref_mut(), residual.read().as_deref());
        x.release();
        let residual = x;

        let x = [residual.clone()];
        let x = ctx.forward(FFN_NORM, ffn_norm, x);
        let x = ctx.forward(FFN_UP, ffn_up, x);
        let x = ctx.forward(FFN_ACT, ffn_act, x);
        let x = ctx.forward(FFN_DOWN, ffn_down, x);

        destruct!([x] = x);
        add(x.write().as_deref_mut(), residual.read().as_deref());
        x.release();

        vec![x]
    }

    fn backward(
        &mut self,
        inputs: impl IntoIterator<Item = RwRc<Tensor<Blob>>>,
        ctx: &mut Context,
    ) -> Vec<RwRc<Tensor<Blob>>> {
        let Self {
            attn_norm,
            attn_qkv,
            attn,
            attn_o,
            ffn_norm,
            ffn_up,
            ffn_act,
            ffn_down,
        } = self;

        destruct!([dresidual] = inputs);
        let dresidual = dresidual;

        let d = [dresidual.clone()];
        let d = ctx.backward(FFN_DOWN, ffn_down, d);
        let d = ctx.backward(FFN_ACT, ffn_act, d);
        let d = ctx.backward(FFN_UP, ffn_up, d);
        let d = ctx.backward(FFN_NORM, ffn_norm, d);

        destruct!([d] = d);
        add(d.write().as_deref_mut(), dresidual.read().as_deref());
        d.release();
        let dresidual = d;

        let d = [dresidual.clone()];
        let d = ctx.backward(ATTN_O, attn_o, d);
        let d = ctx.backward(ATTN, attn, d);
        let d = ctx.backward(ATTN_QKV, attn_qkv, d);
        let d = ctx.backward(ATTN_NORM, attn_norm, d);

        destruct!([d] = d);
        add(d.write().as_deref_mut(), dresidual.read().as_deref());
        d.release();

        vec![d]
    }
}

fn add(y: Tensor<&mut [u8]>, x: Tensor<&[u8]>) {
    assert_eq!(y.shape(), x.shape());
    let ndim = y.layout().ndim();
    let y = y.merge(0, ndim).vector_mut::<f32>();
    let x = x.merge(0, ndim).vector::<f32>();
    for (y, x) in std::iter::zip(y, x) {
        *y += x
    }
}

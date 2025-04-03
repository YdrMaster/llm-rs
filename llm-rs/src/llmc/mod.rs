mod data_loader;
mod tokenizer;

use crate::Tensor;
use digit_layout::types;

pub use data_loader::DataLoader;
pub use tokenizer::{Tokenizer, safe_print};

struct BinHeader([i32; 256]);

/// GPT-2 模型配置
#[derive(Clone, Debug)]
pub struct Gpt2Config {
    pub n_seq: usize,             // 最大序列长度，例如 1024
    pub n_voc: usize,             // 词表大小，例如 50257
    pub padded_vocab_size: usize, // 填充后的词表大小，例如 50304
    pub nblk: usize,              // 层数，例如 12
    pub nh: usize,                // 注意力头数，例如 12
    pub d: usize,                 // 通道数，例如 768
}

pub struct Gpt2<T> {
    pub config: Gpt2Config,
    pub wte: Tensor<T>,
    pub wpe: Tensor<T>,
    pub blks: Box<[Gpt2Blk<T>]>,
    pub output_norm: [Tensor<T>; 2],
}

pub struct Gpt2Blk<T> {
    pub attn_norm: [Tensor<T>; 2],
    pub attn_qkv: [Tensor<T>; 2],
    pub attn_o: [Tensor<T>; 2],
    pub ffn_norm: [Tensor<T>; 2],
    pub ffn_up: [Tensor<T>; 2],
    pub ffn_down: [Tensor<T>; 2],
}

impl<'a> Gpt2<&'a [u8]> {
    pub fn new(data: &'a [u8]) -> Self {
        let (header, mut body) = data.split_at(size_of::<BinHeader>()); // 读取 GPT-2 模型的头部信息
        let header = unsafe { header.as_ptr().cast::<BinHeader>().as_ref().unwrap() };
        if header.0[0] != 20240326 || header.0[1] != 3 {
            panic!("header is not correct");
        }

        // 读取超参数
        let config = Gpt2Config {
            n_seq: header.0[2] as _,
            n_voc: header.0[3] as _,
            padded_vocab_size: header.0[7] as _,
            nblk: header.0[4] as _,
            nh: header.0[5] as _,
            d: header.0[6] as _,
        };

        // 打印模型配置信息
        println!("{config:#?}");

        let Gpt2Config {
            n_seq,
            padded_vocab_size,
            nblk,
            d,
            ..
        } = config;

        let mut tensor = |shape: &[usize]| {
            Tensor::new(types::F32, shape).map(|len| {
                let (data, tail) = body.split_at(len);
                body = tail;
                data
            })
        };

        macro_rules! split {
            ($( $name:ident: $shape:expr )+) => {
                $(
                    let $name = tensor(&$shape);
                )+
            };
        }

        split! {
            wte           : [padded_vocab_size, d]
            wpe           : [n_seq, d]
            attn_norm_w   : [nblk, d]
            attn_norm_b   : [nblk, d]
            attn_qkv_w    : [nblk, 3 * d, d]
            attn_qkv_b    : [nblk, 3 * d,  ]
            attn_o_w      : [nblk, d, d]
            attn_o_b      : [nblk, d,  ]
            ffn_norm_w    : [nblk, d]
            ffn_norm_b    : [nblk, d]
            ffn_up_w      : [nblk, 4 * d, d]
            ffn_up_b      : [nblk, 4 * d   ]
            ffn_down_w    : [nblk, d, 4 * d]
            ffn_down_b    : [nblk, d,      ]
            output_norm_w : [d]
            output_norm_b : [d]
        }

        macro_rules! index {
            ($tensor:ident[$i:expr]) => {
                $tensor.clone().index(&[$i])
            };
        }

        Self {
            config,
            wte,
            wpe,
            blks: (0..nblk)
                .map(|i| Gpt2Blk {
                    attn_norm: [index!(attn_norm_w[i]), index!(attn_norm_b[i])],
                    attn_qkv: [index!(attn_qkv_w[i]), index!(attn_qkv_b[i])],
                    attn_o: [index!(attn_o_w[i]), index!(attn_o_b[i])],
                    ffn_norm: [index!(ffn_norm_w[i]), index!(ffn_norm_b[i])],
                    ffn_up: [index!(ffn_up_w[i]), index!(ffn_up_b[i])],
                    ffn_down: [index!(ffn_down_w[i]), index!(ffn_down_b[i])],
                })
                .collect(),
            output_norm: [output_norm_w, output_norm_b],
        }
    }
}

impl<T> Gpt2<T> {
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Gpt2<U> {
        Gpt2 {
            config: self.config,
            wte: self.wte.map(&mut f),
            wpe: self.wpe.map(&mut f),
            blks: self.blks.into_iter().map(|blk| blk.map(&mut f)).collect(),
            output_norm: self.output_norm.map(|t| t.map(&mut f)),
        }
    }
}

impl<T> Gpt2Blk<T> {
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Gpt2Blk<U> {
        macro_rules! map {
            ($( $id:ident )+) => {
                Gpt2Blk { $( $id: self.$id.map(|t| t.map(&mut f)), )+ }
            };
        }

        map! {
            attn_norm
            attn_qkv
            attn_o
            ffn_norm
            ffn_up
            ffn_down
        }
    }
}

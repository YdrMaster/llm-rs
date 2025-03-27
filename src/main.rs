mod data_loader;
mod fmt;
mod kernel;
mod tensor;
mod tokenizer;

use data_loader::DataLoader;
use digit_layout::types;
use kernel::{add, attention, backward, crossentropy, encode, gelu, layer_norm, mat_mul, softmax};
use memmap2::Mmap;
use std::{fs::File, path::Path, usize};
use tensor::Tensor;
use tokenizer::{Tokenizer, safe_print};

struct BinHeader([i32; 256]);

macro_rules! init {
    ($ty:ident; $( $name:ident: $shape:expr )+) => {
        $ty {
            $(
                $name: Tensor::new(types::F32, $shape)
            ),+
        }
    };
}

macro_rules! map {
    ($ty:ident; $self:expr; $f:expr; $( $name:ident )+) => {
        $ty {
            $(
                $name: $self.$name.map(&mut $f)
            ),+
        }
    };
}

/// GPT-2 模型配置
#[derive(Debug)]
pub struct GPT2Config {
    pub n_seq: usize,             // 最大序列长度，例如 1024
    pub n_voc: usize,             // 词汇表大小，例如 50257
    pub padded_vocab_size: usize, // 填充后的词汇表大小，例如 50304
    pub nblk: usize,              // 层数，例如 12
    pub nh: usize,                // 注意力头数，例如 12
    pub d: usize,                 // 通道数，例如 768
}

impl GPT2Config {
    pub fn new_param(&self) -> ParameterTensors<usize> {
        let &Self {
            n_seq,
            padded_vocab_size,
            nblk,
            d,
            ..
        } = self;
        init! {ParameterTensors;
            wte          : &[padded_vocab_size, d]
            wpe          : &[n_seq, d]
            attn_norm_w  : &[nblk, d]
            attn_norm_b  : &[nblk, d]
            attn_qkv_w   : &[nblk, 3 * d, d]
            attn_qkv_b   : &[nblk, 3 * d]
            attn_o_w     : &[nblk, d, d]
            attn_o_b     : &[nblk, d]
            ffn_norm_w   : &[nblk, d]
            ffn_norm_b   : &[nblk, d]
            ffn_up_w     : &[nblk, 4 * d, d]
            ffn_up_b     : &[nblk, 4 * d]
            ffn_down_w   : &[nblk, d, 4 * d]
            ffn_down_b   : &[nblk, d]
            output_norm_w: &[d]
            output_norm_b: &[d]
        }
    }

    pub fn new_activations(&self, batch_size: usize, seq_len: usize) -> ActivationTensors<usize> {
        let &Self {
            padded_vocab_size,
            nblk,
            d,
            ..
        } = self;
        init! {ActivationTensors;
            embeddings      : &[batch_size, seq_len, d]
            attn_norm       : &[nblk, batch_size, seq_len, d]
            attn_norm_mean  : &[nblk, batch_size, seq_len]
            attn_norm_rstd  : &[nblk, batch_size, seq_len]
            attn_qkv        : &[nblk, batch_size, seq_len, 3 * d]
            attn_out        : &[nblk, batch_size, seq_len, d]
            attn_scores_pre : &[nblk, batch_size, nblk, seq_len, seq_len]
            attn_scores_post: &[nblk, batch_size, nblk, seq_len, seq_len]
            attn_proj       : &[nblk, batch_size, seq_len, d]
            attn_residual   : &[nblk, batch_size, seq_len, d]
            ffn_norm        : &[nblk, batch_size, seq_len, d]
            ffn_norm_mean   : &[nblk, batch_size, seq_len]
            ffn_norm_rstd   : &[nblk, batch_size, seq_len]
            ffn_hidden      : &[nblk, batch_size, seq_len, 4 * d]
            ffn_hidden_gelu : &[nblk, batch_size, seq_len, 4 * d]
            ffn_proj         : &[nblk, batch_size, seq_len, d]
            ffn_residual    : &[nblk, batch_size, seq_len, d]
            output_norm     : &[batch_size, seq_len, d]
            output_norm_mean: &[batch_size, seq_len]
            output_norm_rstd: &[batch_size, seq_len]
            logits          : &[batch_size, seq_len, padded_vocab_size]
            probs           : &[batch_size, seq_len, padded_vocab_size]
            losses          : &[batch_size, seq_len]
        }
    }
}

pub struct ParameterTensors<T> {
    pub wte: Tensor<T>,           // 词嵌入权重，[n_voc, d]
    pub wpe: Tensor<T>,           // 位置嵌入权重，[n_seq, d]
    pub attn_norm_w: Tensor<T>,   // 注意力层归一化权重，[nblk, d]
    pub attn_norm_b: Tensor<T>,   // 注意力层归一化偏置，[nblk, d]
    pub attn_qkv_w: Tensor<T>,    // 注意力 QKV 权重，[nblk, 3*d, d]
    pub attn_qkv_b: Tensor<T>,    // 注意力 QKV 偏置，[nblk, 3*d]
    pub attn_o_w: Tensor<T>,      // 注意力输出投影权重，[nblk, d, d]
    pub attn_o_b: Tensor<T>,      // 注意力输出投影偏置，[nblk, d]
    pub ffn_norm_w: Tensor<T>,    // MLP 层归一化权重，[nblk, d]
    pub ffn_norm_b: Tensor<T>,    // MLP 层归一化偏置，[nblk, d]
    pub ffn_up_w: Tensor<T>,      // MLP 上采样权重，[nblk, 4*d, d]
    pub ffn_up_b: Tensor<T>,      // MLP 上采样偏置，[nblk, 4*d]
    pub ffn_down_w: Tensor<T>,    // MLP 下采样权重，[nblk, d, 4*d]
    pub ffn_down_b: Tensor<T>,    // MLP 下采样偏置，[nblk, d]
    pub output_norm_w: Tensor<T>, // 输出层归一化权重，[d]
    pub output_norm_b: Tensor<T>, // 输出层归一化偏置，[d]
}

impl<T> ParameterTensors<T> {
    fn map<U>(self, mut f: impl FnMut(T) -> U) -> ParameterTensors<U> {
        map! {ParameterTensors; self; f;
            wte
            wpe
            attn_norm_w
            attn_norm_b
            attn_qkv_w
            attn_qkv_b
            attn_o_w
            attn_o_b
            ffn_norm_w
            ffn_norm_b
            ffn_up_w
            ffn_up_b
            ffn_down_w
            ffn_down_b
            output_norm_w
            output_norm_b
        }
    }
}

pub struct ActivationTensors<T> {
    pub embeddings: Tensor<T>,       // 嵌入输出，[B, T, d]
    pub attn_norm: Tensor<T>,        // 注意力层归一化输出，[nblk, B, T, d]
    pub attn_norm_mean: Tensor<T>,   // 注意力层归一化均值，[nblk, B, T]
    pub attn_norm_rstd: Tensor<T>,   // 注意力层归一化反向标准差，[nblk, B, T]
    pub attn_qkv: Tensor<T>,         // 注意力 QKV 输出，[nblk, B, T, 3*d]
    pub attn_out: Tensor<T>,         // 注意力输出，[nblk, B, T, d]
    pub attn_scores_pre: Tensor<T>,  // 注意力分数（预），[nblk, B, nh, T, T]
    pub attn_scores_post: Tensor<T>, // 注意力分数（后），[nblk, B, nh, T, T]
    pub attn_proj: Tensor<T>,        // 注意力投影输出，[nblk, B, T, d]
    pub attn_residual: Tensor<T>,    // 注意力残差连接输出，[nblk, B, T, d]
    pub ffn_norm: Tensor<T>,         // MLP 层归一化输出，[nblk, B, T, d]
    pub ffn_norm_mean: Tensor<T>,    // MLP 层归一化均值，[nblk, B, T]
    pub ffn_norm_rstd: Tensor<T>,    // MLP 层归一化反向标准差，[nblk, B, T]
    pub ffn_hidden: Tensor<T>,       // MLP 隐藏层输出，[nblk, B, T, 4*d]
    pub ffn_hidden_gelu: Tensor<T>,  // MLP 隐藏层 GeLU 激活输出，[nblk, B, T, 4*d]
    pub ffn_proj: Tensor<T>,         // MLP 输出，[nblk, B, T, d]
    pub ffn_residual: Tensor<T>,     // MLP 残差连接输出，[nblk, B, T, d]
    pub output_norm: Tensor<T>,      // 输出层归一化输出，[B, T, d]
    pub output_norm_mean: Tensor<T>, // 输出层归一化均值，[B, T]
    pub output_norm_rstd: Tensor<T>, // 输出层归一化反向标准差，[B, T]
    pub logits: Tensor<T>,           // 逻辑值，[B, T, n_voc]
    pub probs: Tensor<T>,            // 概率分布，[B, T, n_voc]
    pub losses: Tensor<T>,           // 损失值，[B, T]
}

impl<T> ActivationTensors<T> {
    fn map<U>(self, mut f: impl FnMut(T) -> U) -> ActivationTensors<U> {
        map! {ActivationTensors; self; f;
            embeddings
            attn_norm
            attn_norm_mean
            attn_norm_rstd
            attn_qkv
            attn_out
            attn_scores_pre
            attn_scores_post
            attn_proj
            attn_residual
            ffn_norm
            ffn_norm_mean
            ffn_norm_rstd
            ffn_hidden
            ffn_hidden_gelu
            ffn_proj
            ffn_residual
            output_norm
            output_norm_mean
            output_norm_rstd
            logits
            probs
            losses
        }
    }
}

pub struct GPT2 {
    pub config: GPT2Config, // 模型配置

    // 模型参数
    pub params: ParameterTensors<Box<[u8]>>, // 模型权重参数
    pub grads: ParameterTensors<Box<[u8]>>,  // 模型权重参数的梯度

    // 激活值
    pub acts: ActivationTensors<Box<[u8]>>,      // 模型激活值
    pub grad_acts: ActivationTensors<Box<[u8]>>, // 模型激活值的梯度

    // 其他运行时配置
    pub batch_size: usize,          // 当前前向传播的批量大小 (B)
    pub seq_len: usize,             // 当前前向传播的序列长度 (T)
    pub inputs: Tensor<Box<[u8]>>,  // 当前前向传播的输入词元
    pub targets: Tensor<Box<[u8]>>, // 当前前向传播的目标词元
}

impl GPT2 {
    pub fn new(path: impl AsRef<Path>, batch_size: usize, seq_len: usize) -> Self {
        let file = File::open(path).unwrap();
        let mmap = unsafe { Mmap::map(&file).unwrap() };
        let (header, mut body) = mmap.split_at(size_of::<BinHeader>()); // 读取 GPT-2 模型的头部信息
        let header = unsafe { header.as_ptr().cast::<BinHeader>().as_ref().unwrap() };
        if header.0[0] != 20240326 || header.0[1] != 3 {
            panic!("header is not correct");
        }

        // 读取超参数
        let config = GPT2Config {
            n_seq: header.0[2] as _,
            n_voc: header.0[3] as _,
            padded_vocab_size: header.0[7] as _,
            nblk: header.0[4] as _,
            nh: header.0[5] as _,
            d: header.0[6] as _,
        };

        // 打印模型配置信息
        println!("{config:#?}");

        let params = config.new_param().map(|size| {
            let (val, tail) = body.split_at(size);
            body = tail;
            val.iter().copied().collect()
        });
        let grads = config.new_param().map(|size| vec![0u8; size].into());
        let acts = config
            .new_activations(batch_size, seq_len)
            .map(|size| vec![0u8; size].into());
        let grad_acts = config
            .new_activations(batch_size, seq_len)
            .map(|size| vec![0u8; size].into());

        Self {
            config,
            params,
            grads,
            acts,
            grad_acts,
            batch_size,
            seq_len,
            inputs: Tensor::new(types::U16, &[batch_size, seq_len])
                .map(|size| vec![0u8; size].into()),
            targets: Tensor::new(types::U16, &[batch_size, seq_len])
                .map(|size| vec![0u8; size].into()),
        }
    }

    pub fn forward(&mut self, inputs: &[u16], targets: Option<&[u16]>) -> f32 {
        let batch_size = self.batch_size;
        let seq_len = self.seq_len;

        assert_eq!(inputs.len(), seq_len * batch_size);
        self.inputs.as_slice_mut().write(inputs);

        if let Some(targets) = targets {
            assert_eq!(targets.len(), seq_len * batch_size);
            self.targets.as_slice_mut().write(targets);
        }

        let ActivationTensors {
            embeddings,
            attn_norm,
            attn_norm_mean,
            attn_norm_rstd,
            attn_qkv,
            attn_out,
            attn_scores_pre,
            attn_scores_post,
            attn_proj,
            attn_residual,
            ffn_norm,
            ffn_norm_mean,
            ffn_norm_rstd,
            ffn_hidden,
            ffn_hidden_gelu,
            ffn_proj,
            ffn_residual,
            output_norm,
            output_norm_mean,
            output_norm_rstd,
            logits,
            probs,
            losses,
        } = &mut self.acts;

        encode(
            embeddings.as_slice_mut(),
            self.inputs.as_slice(),
            self.params.wte.as_slice(),
            self.params.wpe.as_slice(),
        );
        let mut residual = embeddings.as_slice();
        for iblk in 0..self.config.nblk {
            layer_norm(
                attn_norm.as_slice_mut().index(&[iblk]),
                attn_norm_mean.as_slice_mut().index(&[iblk]),
                attn_norm_rstd.as_slice_mut().index(&[iblk]),
                residual.as_slice(),
                self.params.attn_norm_w.as_slice().index(&[iblk]),
                self.params.attn_norm_b.as_slice().index(&[iblk]),
            );
            mat_mul(
                attn_qkv.as_slice_mut().index(&[iblk]),
                attn_norm.as_slice().index(&[iblk]),
                self.params.attn_qkv_w.as_slice().index(&[iblk]),
                Some(self.params.attn_qkv_b.as_slice().index(&[iblk])),
            );
            attention(
                attn_out.as_slice_mut().index(&[iblk]),
                attn_scores_pre.as_slice_mut().index(&[iblk]),
                attn_scores_post.as_slice_mut().index(&[iblk]),
                attn_qkv.as_slice().index(&[iblk]),
            );
            mat_mul(
                attn_proj.as_slice_mut().index(&[iblk]),
                attn_out.as_slice().index(&[iblk]),
                self.params.attn_o_w.as_slice().index(&[iblk]),
                Some(self.params.attn_o_b.as_slice().index(&[iblk])),
            );
            add(
                attn_residual.as_slice_mut().index(&[iblk]),
                residual.as_slice(),
                attn_proj.as_slice().index(&[iblk]),
            );
            layer_norm(
                ffn_norm.as_slice_mut().index(&[iblk]),
                ffn_norm_mean.as_slice_mut().index(&[iblk]),
                ffn_norm_rstd.as_slice_mut().index(&[iblk]),
                attn_residual.as_slice().index(&[iblk]),
                self.params.ffn_norm_w.as_slice().index(&[iblk]),
                self.params.ffn_norm_b.as_slice().index(&[iblk]),
            );
            mat_mul(
                ffn_hidden.as_slice_mut().index(&[iblk]),
                ffn_norm.as_slice().index(&[iblk]),
                self.params.ffn_up_w.as_slice().index(&[iblk]),
                Some(self.params.ffn_up_b.as_slice().index(&[iblk])),
            );
            gelu(
                ffn_hidden_gelu.as_slice_mut().index(&[iblk]),
                ffn_hidden.as_slice().index(&[iblk]),
            );
            mat_mul(
                ffn_proj.as_slice_mut().index(&[iblk]),
                ffn_hidden_gelu.as_slice().index(&[iblk]),
                self.params.ffn_down_w.as_slice().index(&[iblk]),
                Some(self.params.ffn_down_b.as_slice().index(&[iblk])),
            );
            add(
                ffn_residual.as_slice_mut().index(&[iblk]),
                attn_residual.as_slice().index(&[iblk]),
                ffn_proj.as_slice().index(&[iblk]),
            );
            residual = ffn_residual.as_slice().index(&[iblk]);
        }
        layer_norm(
            output_norm.as_slice_mut(),
            output_norm_mean.as_slice_mut(),
            output_norm_rstd.as_slice_mut(),
            residual,
            self.params.output_norm_w.as_slice(),
            self.params.output_norm_b.as_slice(),
        );
        mat_mul(
            logits.as_slice_mut(),
            output_norm.as_slice(),
            self.params.wte.as_slice(),
            None,
        );
        softmax(probs.as_slice_mut(), logits.as_slice(), self.config.n_voc);

        if targets.is_some() {
            crossentropy(
                losses.as_slice_mut(),
                probs.as_slice(),
                self.targets.as_slice(),
            );
            losses
                .as_slice()
                .merge(0, 2)
                .vector::<f32>()
                .iter()
                .sum::<f32>()
                / (batch_size * seq_len) as f32
        } else {
            -1.
        }
    }

    pub fn zero_grad(&mut self) {
        self.grads = self.config.new_param().map(|size| vec![0u8; size].into());
        self.grad_acts = self
            .config
            .new_activations(self.batch_size, self.seq_len)
            .map(|size| vec![0u8; size].into());
    }

    pub fn backward(&mut self, _mean_loss: f32) {
        let dloss_mean = 1. / (self.batch_size * self.seq_len) as f32;
        self.grad_acts
            .losses
            .as_slice_mut()
            .merge(0, 2)
            .vector_mut::<f32>()
            .fill(dloss_mean);

        backward::crossentropy_softmax(
            self.grad_acts.logits.as_slice_mut(),
            self.grad_acts.losses.as_slice(),
            self.acts.probs.as_slice(),
            self.targets.as_slice(),
        );
        backward::mat_mul(
            self.grad_acts.output_norm.as_slice_mut(),
            self.grads.wte.as_slice_mut(),
            None,
            self.grad_acts.logits.as_slice(),
            self.acts.output_norm.as_slice(),
            self.params.wte.as_slice(),
        );

        let nblk = self.config.nblk;
        backward::layer_norm(
            self.grad_acts
                .ffn_residual
                .as_slice_mut()
                .index(&[nblk - 1]),
            self.grads.output_norm_w.as_slice_mut(),
            self.grads.output_norm_b.as_slice_mut(),
            self.grad_acts.output_norm.as_slice(),
            self.acts.ffn_residual.as_slice().index(&[nblk - 1]),
            self.params.output_norm_w.as_slice(),
            self.acts.output_norm_mean.as_slice(),
            self.acts.output_norm_rstd.as_slice(),
        );

        for iblk in (0..self.config.nblk).rev() {
            backward::add(
                self.grad_acts.attn_residual.as_slice_mut().index(&[iblk]),
                self.grad_acts.ffn_proj.as_slice_mut().index(&[iblk]),
                self.grad_acts.ffn_residual.as_slice().index(&[iblk]),
            );
            backward::mat_mul(
                self.grad_acts.ffn_hidden_gelu.as_slice_mut().index(&[iblk]),
                self.grads.ffn_down_w.as_slice_mut().index(&[iblk]),
                Some(self.grads.ffn_down_b.as_slice_mut().index(&[iblk])),
                self.grad_acts.ffn_proj.as_slice().index(&[iblk]),
                self.acts.ffn_hidden_gelu.as_slice().index(&[iblk]),
                self.params.ffn_down_w.as_slice().index(&[iblk]),
            );
            backward::gelu(
                self.grad_acts.ffn_hidden.as_slice_mut().index(&[iblk]),
                self.acts.ffn_hidden.as_slice().index(&[iblk]),
                self.grad_acts.ffn_hidden_gelu.as_slice().index(&[iblk]),
            );
            backward::mat_mul(
                self.grad_acts.ffn_norm.as_slice_mut().index(&[iblk]),
                self.grads.ffn_up_w.as_slice_mut().index(&[iblk]),
                Some(self.grads.ffn_up_b.as_slice_mut().index(&[iblk])),
                self.grad_acts.ffn_hidden.as_slice().index(&[iblk]),
                self.acts.ffn_norm.as_slice().index(&[iblk]),
                self.params.ffn_up_w.as_slice().index(&[iblk]),
            );
            backward::layer_norm(
                self.grad_acts.attn_residual.as_slice_mut().index(&[iblk]),
                self.grads.ffn_norm_w.as_slice_mut().index(&[iblk]),
                self.grads.ffn_norm_b.as_slice_mut().index(&[iblk]),
                self.grad_acts.ffn_norm.as_slice().index(&[iblk]),
                self.acts.attn_residual.as_slice().index(&[iblk]),
                self.params.ffn_norm_w.as_slice().index(&[iblk]),
                self.acts.ffn_norm_mean.as_slice().index(&[iblk]),
                self.acts.ffn_norm_rstd.as_slice().index(&[iblk]),
            );

            let (residual, mut dresidual) = if iblk == 0 {
                (
                    self.acts.embeddings.as_slice(),
                    self.grad_acts.embeddings.as_slice_mut(),
                )
            } else {
                (
                    self.acts.ffn_residual.as_slice().index(&[iblk - 1]),
                    self.grad_acts
                        .ffn_residual
                        .as_slice_mut()
                        .index(&[iblk - 1]),
                )
            };

            backward::add(
                dresidual.as_slice_mut(),
                self.grad_acts.attn_proj.as_slice_mut().index(&[iblk]),
                self.grad_acts.attn_residual.as_slice().index(&[iblk]),
            );
            backward::mat_mul(
                self.grad_acts.attn_out.as_slice_mut().index(&[iblk]),
                self.grads.attn_o_w.as_slice_mut().index(&[iblk]),
                Some(self.grads.attn_o_b.as_slice_mut().index(&[iblk])),
                self.grad_acts.attn_proj.as_slice().index(&[iblk]),
                self.acts.attn_out.as_slice().index(&[iblk]),
                self.params.attn_o_w.as_slice().index(&[iblk]),
            );
            backward::attention(
                self.grad_acts.attn_qkv.as_slice_mut().index(&[iblk]),
                self.grad_acts.attn_scores_pre.as_slice_mut().index(&[iblk]),
                self.grad_acts
                    .attn_scores_post
                    .as_slice_mut()
                    .index(&[iblk]),
                self.grad_acts.attn_out.as_slice().index(&[iblk]),
                self.acts.attn_qkv.as_slice().index(&[iblk]),
                self.acts.attn_scores_post.as_slice().index(&[iblk]),
            );
            backward::mat_mul(
                self.grad_acts.attn_norm.as_slice_mut().index(&[iblk]),
                self.grads.attn_qkv_w.as_slice_mut().index(&[iblk]),
                Some(self.grads.attn_qkv_b.as_slice_mut().index(&[iblk])),
                self.grad_acts.attn_qkv.as_slice().index(&[iblk]),
                self.acts.attn_norm.as_slice().index(&[iblk]),
                self.params.attn_qkv_w.as_slice().index(&[iblk]),
            );
            backward::layer_norm(
                dresidual.as_slice_mut(),
                self.grads.attn_norm_w.as_slice_mut().index(&[iblk]),
                self.grads.attn_norm_b.as_slice_mut().index(&[iblk]),
                self.grad_acts.attn_norm.as_slice().index(&[iblk]),
                residual,
                self.params.attn_norm_w.as_slice().index(&[iblk]),
                self.acts.attn_norm_mean.as_slice().index(&[iblk]),
                self.acts.attn_norm_rstd.as_slice().index(&[iblk]),
            )
        }
        backward::encode(
            self.grads.wte.as_slice_mut(),
            self.grads.wpe.as_slice_mut(),
            self.grad_acts.embeddings.as_slice(),
            self.inputs.as_slice(),
        )
    }

    pub fn update(
        &mut self,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        step: usize,
    ) {
    }
}

fn main() {
    let batch_size = 4;
    let seq_len = 64;
    let mut gpt2 = GPT2::new(
        "/home/mechdancer/repos/llm.c/gpt2_124M.bin",
        batch_size,
        seq_len,
    );
    let mut train_loader = DataLoader::new(
        "/home/mechdancer/repos/llm.c",
        "*/tiny_shakespeare_train.bin",
        batch_size,
        seq_len,
        false,
    );
    let mut val_loader = DataLoader::new(
        "/home/mechdancer/repos/llm.c",
        "*/tiny_shakespeare_val.bin",
        batch_size,
        seq_len,
        false,
    );
    let tokenizer = Tokenizer::new("/home/mechdancer/repos/llm.c/gpt2_tokenizer.bin").unwrap();
    assert_eq!(tokenizer.decode(tokenizer.eos), br"<|endoftext|>");

    for step in 0..=40 {
        if step % 10 == 0 {
            val_loader.rand();
            let mut val_loss = 0.;
            for _ in 0..5 {
                let [inputs, targets] = val_loader.load();
                val_loss += gpt2.forward(inputs, Some(targets));
            }
            val_loss /= 5.;
            println!("step: {step}, val_loss: {val_loss}");
        }

        if step > 0 && step % 20 == 0 {
            println!("-----------");
            let mut gen_tokens = vec![tokenizer.eos; batch_size * seq_len];
            for t in 1..64 {
                gpt2.forward(&gen_tokens, None);
                let probs = gpt2
                    .acts
                    .probs
                    .as_slice()
                    .index(&[0, t - 1])
                    .vector::<f32>();
                let next = sample(probs, rand::random());
                gen_tokens[t] = next as u16;
                safe_print(tokenizer.decode(next as u16));
            }
            println!();
            println!("-----------");
        }

        let [inputs, targets] = train_loader.load();
        let mean_loss = gpt2.forward(inputs, Some(targets));
        gpt2.zero_grad();
        gpt2.backward(mean_loss);
        gpt2.update(1e-4, 0.9, 0.999, 1e-8, 0., step + 1);
    }
}

fn sample(probs: &[f32], coin: f32) -> u16 {
    let mut acc = 0.;
    for (i, p) in probs.iter().enumerate() {
        acc += p;
        if acc >= coin {
            return i as _;
        }
    }
    probs.len() as u16 - 1
}

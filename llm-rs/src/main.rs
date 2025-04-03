mod blob;
mod context;
mod llmc;
mod nn;

use blob::Blob;
use context::Context;
use digit_layout::types;
use llmc::{DataLoader, Tokenizer, safe_print};

type Tensor<T> = tensor::Tensor<T, 4>;

fn main() {
    use memmap2::Mmap;
    use std::fs::File;

    let batch_size = 4;
    let seq_len = 64;

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

    let file = File::open("/home/mechdancer/repos/llm.c/gpt2_124M.bin").unwrap();
    let mmap = unsafe { Mmap::map(&file) }.unwrap();
    let gpt2 = llmc::Gpt2::new(&mmap);
    let n_voc = gpt2.config.n_voc;

    let mut ctx = Context::new();
    let mut gpt2 = ctx.init::<nn::gpt2::Gpt2>("gpt2", gpt2.map(Blob::from));
    let mut loss = ctx.init::<nn::loss::Loss>("loss", n_voc);

    for step in 0..=40 {
        if step % 10 == 0 {
            val_loader.rand();
            let mut val_loss = 0.;
            for _ in 0..5 {
                let [inputs, targets] = val_loader.load();

                let shape = [batch_size, seq_len];
                let tokens = Tensor::new(types::U16, &shape).map(|_| inputs.into());
                let targets = Tensor::new(types::U16, &shape).map(|_| targets.into());

                let logits = ctx.forward("gpt2", &mut gpt2, [tokens.share()]);
                let losses = ctx.forward("loss", &mut loss, [logits[0].clone(), targets.share()]);

                val_loss += losses[0]
                    .read()
                    .as_deref()
                    .merge(0, 2)
                    .vector::<f32>()
                    .iter()
                    .sum::<f32>()
                    / (batch_size * seq_len) as f32
            }
            val_loss /= 5.;
            println!("step: {step}, val_loss: {val_loss}");
        }

        if step > 0 && step % 20 == 0 {
            println!("-----------");
            let mut tokens = vec![tokenizer.eos; batch_size * seq_len];
            for t in 1..64 {
                let tokens_ =
                    Tensor::new(types::U16, &[batch_size, seq_len]).map(|_| Blob::from(&*tokens));
                let logits = ctx.forward("gpt2", &mut gpt2, [tokens_.share()]);
                let logits = logits[0].read().as_deref().index(&[0, t - 1]).vector();
                let next = sample(&logits[..n_voc], rand::random());
                tokens[t] = next as u16;
                safe_print(tokenizer.decode(next as u16))
            }
            println!();
            println!("-----------");
        }

        let [inputs, targets] = train_loader.load();

        let shape = [batch_size, seq_len];
        let tokens = Tensor::new(types::U16, &shape).map(|_| inputs.into());
        let targets = Tensor::new(types::U16, &shape).map(|_| targets.into());

        let logits = ctx.forward("gpt2", &mut gpt2, [tokens.share()]);
        let losses = ctx.forward("loss", &mut loss, [logits[0].clone(), targets.share()]);
        ctx.zero_grad();

        let dloss_mean = 1. / (batch_size * seq_len) as f32;
        let mut dlosses = Tensor::contiguous_of(losses[0].read()).map(Blob::new);
        dlosses
            .as_deref_mut()
            .merge(0, 2)
            .vector_mut::<f32>()
            .fill(dloss_mean);

        let dlogits = ctx.backward("loss", &mut loss, [dlosses.share()]);
        let _ = ctx.backward("gpt2", &mut gpt2, dlogits);
        // gpt2.update(1e-4, 0.9, 0.999, 1e-8, 0., step + 1);
    }
}

fn sample(logits: &[f32], coin: f32) -> u16 {
    let mut pairs = logits.iter().copied().enumerate().collect::<Vec<_>>();
    pairs.sort_by(|(_, a), (_, b)| f32::total_cmp(a, b).reverse());

    let max = pairs[0].1;
    pairs[0].1 = 1.;

    for i in 1..pairs.len() {
        pairs[i].1 = pairs[i - 1].1 + (pairs[i].1 - max).exp()
    }

    let &[.., (_, sum)] = &*pairs else {
        unreachable!()
    };

    let plimit = sum * coin;
    for (i, acc) in pairs {
        if acc >= plimit {
            return i as _;
        }
    }
    unreachable!()
}

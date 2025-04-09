mod blob;
mod context;
mod llmc;
mod nn;
mod op;
mod optimizer;

use std::{hash::Hash, rc::Weak};

use blob::Blob;
use context::Context;
use rw_rc::RwRc;

type Tensor<T> = tensor::Tensor<T, 4>;

fn main() {
    use digit_layout::types;
    use llmc::{DataLoader, Tokenizer, safe_print};
    use memmap2::Mmap;
    use optimizer::AdamW;
    use std::fs::File;
    use std::{env::args, path::PathBuf, time::Instant};

    let bin_path = PathBuf::from(args().nth(1).unwrap());
    let batch_size = 4;
    let seq_len = 64;

    let mut train_loader = DataLoader::new(
        &bin_path,
        "*/tiny_shakespeare_train.bin",
        batch_size,
        seq_len,
        true,
    );
    let mut val_loader = DataLoader::new(
        &bin_path,
        "*/tiny_shakespeare_val.bin",
        batch_size,
        seq_len,
        false,
    );
    let tokenizer = Tokenizer::new(bin_path.join("gpt2_tokenizer.bin")).unwrap();
    assert_eq!(tokenizer.decode(tokenizer.eos), br"<|endoftext|>");

    let file = File::open(bin_path.join("gpt2_124M.bin")).unwrap();
    let mmap = unsafe { Mmap::map(&file) }.unwrap();
    let gpt2 = llmc::Gpt2::new(&mmap);
    let n_voc = gpt2.config.n_voc;

    let mut ctx = Context::new(false);
    let mut gpt2 = ctx.init::<nn::gpt2::Gpt2>("gpt2", gpt2.map(Blob::from).map(RwRc::new));
    let mut loss = ctx.init::<nn::loss::Loss>("loss", n_voc);
    let mut adamw = AdamW::new(1e-4, 0.9, 0.999, 1e-8, 0.);

    for step in 0..=40 {
        if step % 10 == 0 {
            val_loader.rand();
            let mut val_loss = 0.;
            for _ in 0..5 {
                let [inputs, targets] = val_loader.load();

                let shape = [batch_size, seq_len];
                let tokens = Tensor::new(types::U16, &shape).map(|_| RwRc::new(inputs.into()));
                let targets = Tensor::new(types::U16, &shape).map(|_| RwRc::new(targets.into()));

                let logits = ctx.forward("gpt2", &mut gpt2, [tokens.share()]);
                let losses = ctx.forward("loss", &mut loss, [logits[0].clone(), targets.share()]);

                val_loss += loss_sum(losses[0].cloned().as_ref().map(|b| &**b.read()))
            }
            val_loss /= 5.;
            println!("val_loss: {val_loss}")
        }

        if step > 0 && step % 20 == 0 {
            println!("-----------");
            let mut tokens = vec![tokenizer.eos; batch_size * seq_len];
            for t in 1..64 {
                let tokens_ = Tensor::new(types::U16, &[batch_size, seq_len])
                    .map(|_| Blob::from(&*tokens))
                    .map(RwRc::new);
                let logits = ctx.forward("gpt2", &mut gpt2, [tokens_.share()]);
                let logits = logits[0].cloned().index(&[0, t - 1]);
                let logits = logits.as_ref().map(|b| &**b.read()).vector();
                let next = sample(&logits[..n_voc], rand::random());
                tokens[t] = next as u16;
                safe_print(tokenizer.decode(next as u16))
            }
            println!();
            println!("-----------")
        }

        let time = Instant::now();

        let [inputs, targets] = train_loader.load();

        let shape = [batch_size, seq_len];
        let tokens = Tensor::new(types::U16, &shape).map(|_| RwRc::new(inputs.into()));
        let targets = Tensor::new(types::U16, &shape).map(|_| RwRc::new(targets.into()));

        let logits = ctx.forward("gpt2", &mut gpt2, [tokens.share()]);
        let losses = ctx.forward("loss", &mut loss, [logits[0].clone(), targets.share()]);
        let train_loss = loss_sum(losses[0].cloned().as_ref().map(|b| &**b.read()));
        ctx.zero_grad();

        let dloss_mean = 1. / (batch_size * seq_len) as f32;
        let loss_ = &losses[0];
        let dlosses = ctx.tensor(loss_.dt(), &loss_.shape());
        dlosses
            .cloned()
            .merge(0, 2)
            .as_ref()
            .map(|b| &mut **b.write())
            .vector_mut::<f32>()
            .fill(dloss_mean);

        let dlogits = ctx.backward("loss", &mut loss, [dlosses.share()]);
        let _ = ctx.backward("gpt2", &mut gpt2, dlogits);
        ctx.update(&mut adamw);
        adamw.next();

        println!(
            "step {step}: train loss {train_loss} (took {:?})",
            time.elapsed()
        )
    }
}

fn loss_sum(losses: Tensor<&[u8]>) -> f32 {
    let losses = losses.merge(0, 2).vector::<f32>();
    losses.iter().sum::<f32>() / losses.len() as f32
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

struct HashWeak<T>(Weak<T>);

impl<T> PartialEq for HashWeak<T> {
    fn eq(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.0, &other.0)
    }
}

impl<T> Eq for HashWeak<T> {}

impl<T> Hash for HashWeak<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.as_ptr().hash(state)
    }
}

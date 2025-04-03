mod blob;
mod context;
mod llmc;
mod nn;

use blob::Blob;
use context::Context;
use digit_layout::types;
use llmc::{DataLoader, Tokenizer};

type Tensor<T> = tensor::Tensor<T, 4>;

fn main() {
    use memmap2::Mmap;
    use std::fs::File;

    let batch_size = 4;
    let seq_len = 64;

    let _train_loader = DataLoader::new(
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

    let mut ctx = Context::new();
    let mut gpt2 = ctx.init::<nn::gpt2::Gpt2>("gpt2", gpt2.map(Blob::from));

    for step in 0..=40 {
        if step % 10 == 0 {
            val_loader.rand();
            let mut val_loss = 0.;
            for _ in 0..5 {
                let [inputs, _targets] = val_loader.load();
                let tokens = Tensor::new(types::U16, &[batch_size, seq_len]).map(|_| inputs.into());
                let _y = ctx.forward("gpt2", &mut gpt2, vec![tokens.share()]);

                // println!("{}", _y[0].read().as_deref());
                // std::process::exit(0);

                // val_loss += gpt2.forward(inputs, Some(targets));
            }
            val_loss /= 5.;
            println!("step: {step}, val_loss: {val_loss}");
        }

        if step > 0 && step % 20 == 0 {
            println!("-----------");
            // let mut gen_tokens = vec![tokenizer.eos; batch_size * seq_len];
            //     for t in 1..64 {
            //         gpt2.forward(&gen_tokens, None);
            //         let probs = gpt2
            //             .acts
            //             .probs
            //             .as_slice()
            //             .index(&[0, t - 1])
            //             .vector::<f32>();
            //         let next = sample(probs, rand::random());
            //         gen_tokens[t] = next as u16;
            //         safe_print(tokenizer.decode(next as u16));
            //     }
            println!();
            println!("-----------");
        }

        // let [inputs, targets] = train_loader.load();
        // let mean_loss = gpt2.forward(inputs, Some(targets));
        // gpt2.zero_grad();
        // ctx.backward("gpt2", &gpt2, vec![]);
        // gpt2.update(1e-4, 0.9, 0.999, 1e-8, 0., step + 1);
    }
}

use super::BinHeader;
use globset::Glob;
use memmap2::Mmap;
use rand::seq::SliceRandom;
use std::{fs::File, path::Path};

pub struct DataLoader {
    shards: Vec<Shard>,
    batch_size: usize,
    seq_len: usize,
    should_shuffle: bool,
    rng: rand::rngs::ThreadRng,
}

struct Shard {
    tokens: Vec<u16>,
    indices: Vec<usize>,
    sample_idx: usize,
}

impl DataLoader {
    pub fn new(
        path: impl AsRef<Path>,
        pattern: &str,
        batch_size: usize,
        seq_len: usize,
        should_shuffle: bool,
    ) -> Self {
        let glob = Glob::new(pattern).unwrap().compile_matcher();
        let mut shards = vec![];
        for_files(path, &mut |path| {
            if glob.is_match(path) {
                let tokens = load_shard(path);
                let samples = tokens.len() / (batch_size * seq_len);
                println!(
                    "loaded {} tokens ({samples} samples) from {}",
                    tokens.len(),
                    path.display()
                );
                shards.push(Shard {
                    tokens,
                    indices: (0..samples).collect(),
                    sample_idx: 0,
                })
            }
        });

        Self {
            shards,
            batch_size,
            seq_len,
            should_shuffle,
            rng: rand::rng(),
        }
    }

    pub fn rand(&mut self) {
        if self.should_shuffle {
            for Shard { indices, .. } in &mut self.shards {
                indices.shuffle(&mut self.rng)
            }
        }
    }

    pub fn load(&mut self) -> [&[u16]; 2] {
        let n_tok = self.batch_size * self.seq_len;
        let shard = &mut self.shards[0];
        let slice = &shard.tokens[shard.sample_idx * n_tok..];
        shard.sample_idx += 1;
        if shard.sample_idx == shard.indices.len() {
            shard.sample_idx = 0;
        }
        [&slice[..n_tok], &slice[1..][..n_tok]]
    }
}

fn load_shard(path: impl AsRef<Path>) -> Vec<u16> {
    let file = File::open(path).unwrap();
    let mmap = unsafe { Mmap::map(&file).unwrap() };
    let (header, body) = mmap.split_at(size_of::<BinHeader>()); // 读取 GPT-2 模型的头部信息
    let header = unsafe { header.as_ptr().cast::<BinHeader>().as_ref().unwrap() };
    if header.0[0] != 20240520 || header.0[1] != 1 {
        panic!("header is not correct ");
    }
    let ntok = header.0[2] as usize;
    let ([], tokens, []) = (unsafe { body.align_to::<u16>() }) else {
        unreachable!()
    };
    assert_eq!(tokens.len(), ntok);
    tokens.to_vec()
}

fn for_files(path: impl AsRef<Path>, f: &mut impl FnMut(&Path)) {
    path.as_ref()
        .read_dir()
        .unwrap()
        .filter_map(|dir| dir.ok())
        .for_each(|dir| {
            let ty = dir.file_type().unwrap();
            if ty.is_dir() {
                for_files(dir.path(), f)
            } else if ty.is_file() || ty.is_symlink() {
                f(&dir.path())
            }
        })
}

#[test]
fn test_glob() {
    let matcher = globset::Glob::new("./src/*.rs").unwrap().compile_matcher();
    for_files(".", &mut |file| {
        if matcher.is_match(file.to_str().unwrap()) {
            println!("{}", file.display());
        }
    })
}

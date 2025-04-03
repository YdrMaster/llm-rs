use super::BinHeader;
use memmap2::Mmap;
use std::{fs::File, io::Write, path::Path};

// 定义分词器结构体
pub struct Tokenizer {
    token_table: Vec<Vec<u8>>,
    pub eos: u16, // <|endoftext|> token id
}

impl Tokenizer {
    // 初始化分词器
    pub fn new(path: impl AsRef<Path>) -> Result<Tokenizer, std::io::Error> {
        let file = File::open(path).unwrap();
        let mmap = unsafe { Mmap::map(&file).unwrap() };
        let (header, mut body) = mmap.split_at(size_of::<BinHeader>());
        let header = unsafe { header.as_ptr().cast::<BinHeader>().as_ref().unwrap() };
        if header.0[0] != 20240328 {
            panic!("header is not correct ");
        }

        let version = header.0[1];
        let n_voc = header.0[2] as usize;

        let eos = match version {
            1 => {
                assert_eq!(n_voc, 50257);
                50256
            }
            2 => header.0[3] as _,
            _ => panic!("tokenizer version is not supported"),
        };

        // 读取所有tokens
        let mut token_table = Vec::with_capacity(n_voc);
        for _ in 0..n_voc {
            let [len, tail @ ..] = body else {
                panic!("meet EOF while reading token table")
            };
            let len = *len as usize;
            assert!(len > 0); // 每个token至少是一个字符
            let (token_bytes, tail) = tail.split_at(len);
            token_table.push(token_bytes.to_vec());
            body = tail;
        }

        Ok(Tokenizer { token_table, eos })
    }

    // 解码token id
    pub fn decode(&self, token_id: u16) -> &[u8] {
        &self.token_table[token_id as usize]
    }
}

// 安全打印函数
pub fn safe_print(piece: &[u8]) {
    if let Ok(s) = std::str::from_utf8(piece) {
        print!("{s}")
    } else {
        print!("{piece:02x?}")
    }
    std::io::stdout().flush().unwrap()
}

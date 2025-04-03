# llm.c 的 Rust 版本 \[N\]

[![CI](https://github.com/YdrMaster/llm-rs/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/YdrMaster/llm-rs/actions)
[![license](https://img.shields.io/github/license/YdrMaster/llm-rs)](https://mit-license.org/)
[![GitHub Issues](https://img.shields.io/github/issues/YdrMaster/llm-rs)](https://github.com/YdrMaster/llm-rs/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/YdrMaster/llm-rs)](https://github.com/YdrMaster/llm-rs/pulls)
![GitHub repo size](https://img.shields.io/github/repo-size/YdrMaster/llm-rs)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/YdrMaster/llm-rs)
![GitHub contributors](https://img.shields.io/github/contributors/YdrMaster/llm-rs)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/YdrMaster/llm-rs)

先用 llm.c 的脚本下载训练数据，然后：

```shell
cargo run --release -- `<llm.c>`
```

`<llm.c>` 是 llm.c 下载训练集的路径，通常是 llm.c 的项目目录。

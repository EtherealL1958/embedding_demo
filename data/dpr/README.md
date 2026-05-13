# DPR-NQ 数据下载说明

本目录用于存放实验一所需的 DPR-NQ 检索训练与评估数据。

请在本目录下下载以下三个文件：

```bash
cd data/dpr

wget -O biencoder-nq-dev.json.gz \
https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz

wget -O biencoder-nq-train.json.gz \
https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz

wget -O biencoder-nq-adv-hn-train.json.gz \
https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-adv-hn-train.json.gz
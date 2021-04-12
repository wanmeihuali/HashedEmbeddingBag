# HashedEmbeddingBag

This is a sub-project for [dlrm_ssm](https://github.com/yanzhoupan/dlrm_ssm). 
It applies the idea from the paper 
[Compressing Neural Netwkrs with the Hashing Trick](https://arxiv.org/pdf/1504.04788.pdf) to the Embedding Bags in Pytorch.

## How to install

### pre-requests

The project requests cudatoolkit-dev to be compiled.
If you installed CUDA Toolkit from Nvidia's official website, then it should be fine.

If you installed CUDA Toolkit from conda, my suggestion is to create a separate environment and install cudatoolkit-dev by:
```
conda install -c conda-forge cudatoolkit-dev
```

### install
First, clone the repository:
```
git clone https://github.com/wanmeihuali/HashedEmbeddingBag.git
```

Then, goes into the repository directory:
```
cd HashedEmbeddingBag
```

And install the package:
```
python setup.py install
```

## How to use

The API is similar to The [EmbeddingBag API](https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html) of Pytorch.

### Parameters
For HashedEmbeddingBag: 

|name|explanation|
|---|---|
|num_embeddings (int) | size of the dictionary of embeddings |
|embedding_dim (int) | the size of each embedding vector |
|compression(float) | the ratio between the size of embedding for HashedEmbeddingBag and the size of embedding for Pytorch's EmbeddingBag|
|mode(string) | only the sum mode is supported now |
|_weight(Tensor) | A one dimension Tensor used as the hashing buffer, if _weight is provided, the compression parameter will be ignored |

For forwarding:

- input and offsets have to be of the same type, either int or long.
- if input is 2D of shape (B, N),
  - it will be treated as B bags (sequences) each of fixed length N, and this will return B values aggregated in a way depending on the mode. offsets is ignored and required to be None in this case.
- If input is 1D of shape (N), 
    - it will be treated as a concatenation of multiple bags (sequences). offsets is required to be a 1D tensor containing the starting index positions of each bag in input. Therefore, for offsets of shape (B), input will be viewed as having B bags. Empty bags (i.e., having 0-length) will have returned vectors filled by zeros.

### Example
if we want to embedding a category with 1000 different values, and  the length of each embedding is 16.
```python
num_categories = 1000
embedding_dim = 16
compression = 0.1
EE = hashedEmbeddingBag.HashedEmbeddingBag(num_categories, embedding_dim, compression, "sum")

```
If we have a input contains 18 bags, each bag contains 0 - 7 category values.
```python

bag_num = 18

bag_size = torch.randint(low=0, high=7, size=(bag_num,))
indices_num = bag_size.sum().item()

indices = torch.randint(low=0, high=num_categories - 1, size=(indices_num,))
offsets = torch.cat([torch.zeros(1, dtype=torch.long), bag_size.cumsum(dim=0)[:-1]])
```
To do the Embedding, run:
```python
embeddings = EE(indices, offsets)
```
The embeddings will be a 18 x 16 Tensor.

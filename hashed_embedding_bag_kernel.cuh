#include <torch/torch.h>


#include <cuda.h>
#include <cuda_runtime.h>


std::tuple <torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> hashed_embedding_bag_forward(
        const torch::Tensor &hashed_weights,
        const torch::Tensor &indices,
        const torch::Tensor &offsets,
        //const bool scale_grad_by_freq,
        const int64_t mode,
        const int64_t embedding_dim);


torch::Tensor hashed_embedding_bag_backward(
        const torch::Tensor &grad,
        const torch::Tensor &indices,
        const torch::Tensor &offsets,
        const torch::Tensor &offset2bag,
        const torch::Tensor &bag_size_,
        const torch::Tensor &max_indices_,
        const torch::Tensor &hashed_index_,
        int64_t num_weights,
        bool scale_grad_by_freq,
        int64_t mode,
        int64_t embedding_dim);


__device__ __host__ int64_t hash_func(int64_t a, int64_t b);

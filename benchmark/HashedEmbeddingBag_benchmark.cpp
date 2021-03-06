//
// Created by kuangyuan on 4/25/21.
//
#include "hashed_embedding_bag_kernel.cuh"
#include <cstdio>
#include <chrono>

const int running_num = 1000;
const bool run_original_code = false;
const std::vector<std::vector<int>> input_sizes {
        {0, 2048, 20000, 16, 10000},
        {1, 2048, 20000, 16, 10000},
        {2, 2048, 20000, 16, 10000},
        {3, 2048, 20000, 16, 10000}
};

std::string int2Mode(int mode)
{
    switch (mode) {
        case 0:
            return "sum";
        case 1:
            return "mean";
        case 2:
            return "max";
        case 3:
            return "single";
        default:
            return "";
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> generateData(
        int mode, int bag_num, int num_categories, int num_feature, int hashed_weight_size)
{

    auto long_cuda_option = torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA);

    auto hashed_weights = torch::rand({hashed_weight_size}, torch::kCUDA);
    auto weights = torch::rand({num_categories, num_feature}, torch::kCUDA);

    auto bag_size = torch::randint(0, 7, {bag_num}, long_cuda_option);
    if (mode == 3) {
        bag_size = torch::ones({bag_num}, long_cuda_option);
    }
    int indices_num = bag_size.sum().item().toInt();
    auto indices = torch::randint(0, num_categories - 1, {indices_num}, long_cuda_option);
    auto offsets = torch::cat({
                                      torch::zeros(1, long_cuda_option),
                                      bag_size.cumsum(0).slice(0, 0, -1)});


    auto output_grad = torch::rand({bag_num, num_feature}, torch::kCUDA);
    return {indices, offsets, hashed_weights, weights, bag_size, output_grad};
}

void runEmbeddingForwardAndBackward(
        int mode,
        int num_feature,
        torch::Tensor& indices,
        torch::Tensor& offsets,
        torch::Tensor& weight,
        torch::Tensor& bag_size,
        torch::Tensor& output_grad)
{
    if (mode == 3)
    {
        auto output = torch::embedding(weight, indices);
        torch::embedding_backward(output_grad, indices, weight.size(0), 0, false, false);
    }
    else
    {
        // run forward function on GPU
        auto [output, offset2bag, bag_size_generated, max_indices] =
        torch::embedding_bag(weight, indices, offsets, false, mode);

        torch::_embedding_bag_backward(
                output_grad,
                indices,
                offsets,
                offset2bag,
                bag_size_generated,
                max_indices,
                weight.size(0),
                false,
                mode,
                false,
                // I don't find a way to create an undefined tensor, the grad of a tensor is undefined by default.
                indices.grad());

    }

    cudaDeviceSynchronize();
}

void runForwardAndBackward(
        int mode,
        int num_feature,
        torch::Tensor& indices,
        torch::Tensor& offsets,
        torch::Tensor& hashed_weights,
        torch::Tensor& bag_size,
        torch::Tensor& output_grad)
{

    // run forward function on GPU
    auto [output, offset2bag, bag_size_generated, max_indices, hashed_idx] =
        hashed_embedding_bag_forward(hashed_weights, indices, offsets, mode, num_feature);


    auto weight_grad = hashed_embedding_bag_backward(
            output_grad, indices, offsets, offset2bag, bag_size_generated, max_indices, hashed_idx, hashed_weights.size(0), false,
            mode, num_feature);

    cudaDeviceSynchronize();
}




void runBenchmark()
{

    for (auto& config : input_sizes) {
        int mode = config[0];
        int bag_num = config[1];
        int num_categories = config[2];
        int num_feature = config[3];
        int hashed_weight_size = config[4];
        auto [indices, offsets, hashed_weights, weights, bag_size, output_grad] =
            generateData(mode, bag_num, num_categories, num_feature, hashed_weight_size);

        auto start = std::chrono::high_resolution_clock::now();
        for (int idx = 0; idx < running_num; ++idx) {
            runForwardAndBackward(
                    mode,
                    num_feature,
                    indices,
                    offsets,
                    hashed_weights,
                    bag_size,
                    output_grad);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = end - start;
        auto duration_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
        double avg_running_time = static_cast<double>(duration_microseconds) / running_num;
        std::string mode_str = int2Mode(mode);
        printf("Test group: mode=%s, bag_num=%d, num_categories=%d, num_feature=%d, hashed_weight_size=%d\n",
               mode_str.c_str(),
               bag_num,
               num_categories,
               num_feature,
               hashed_weight_size);
        printf(">> avg running time %f microseconds\n", avg_running_time);
        if (run_original_code)
        {
            start = std::chrono::high_resolution_clock::now();
            for (int idx = 0; idx < running_num; ++idx) {
                runEmbeddingForwardAndBackward(
                        mode,
                        num_feature,
                        indices,
                        offsets,
                        weights,
                        bag_size,
                        output_grad);
            }

            end = std::chrono::high_resolution_clock::now();
            duration = end - start;
            duration_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
            avg_running_time = static_cast<double>(duration_microseconds) / running_num;

            printf(">> original embedding bag avg running time %f microseconds\n", avg_running_time);
        }

    }


}


int main()
{
    runBenchmark();
    return 0;
}





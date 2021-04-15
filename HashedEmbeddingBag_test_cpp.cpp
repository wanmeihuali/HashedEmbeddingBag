//
// Created by kuangyuan on 4/13/21.
//
#include "hashed_embedding_bag_kernel.cuh"
#include "gtest/gtest.h"
#include <iostream>


torch::Tensor make_offset2bag(torch::Tensor& offsets, torch::Tensor& indices) {
    auto offsets2bag = torch::zeros(indices.size(0) + 1, indices.options());
    offsets2bag.index_add_(0, offsets, torch::ones_like(offsets).contiguous());
    offsets2bag[0] -= 1;
    offsets2bag = offsets2bag.cumsum(0);
    offsets2bag.resize_(indices.size(0));
    return offsets2bag;
}


TEST(hashed_embedding_bag_kernel, sum_mode) {
    int mode = 0;
    int bag_num = 18;
    int num_categories = 100;
    int num_feature = 200;
    int hashed_weight_size = 200;


    auto long_cuda_option = torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA);

    auto hashed_weights = torch::rand({hashed_weight_size}, torch::kCUDA);
    auto bag_size = torch::randint(0, 7, {bag_num}, long_cuda_option);
    int indices_num = bag_size.sum().item().toInt();
    auto indices = torch::randint(0, num_categories - 1, {indices_num}, long_cuda_option);
    auto offsets = torch::cat({
        torch::zeros(1, long_cuda_option),
        bag_size.cumsum(0).slice(0, 0, -1)});


    // run forward function on GPU
    auto [output, offset2bag, bag_size_generated, max_indices, hashed_idx] =
        hashed_embedding_bag_forward(hashed_weights, indices, offsets, mode, num_feature);

    {
        auto device = torch::kCPU;
        hashed_weights = hashed_weights.to(device);
        indices = indices.to(device);
        offsets = offsets.to(device);
        output = output.to(device);
        offset2bag = offset2bag.to(device);
        bag_size = bag_size.to(device);
        bag_size_generated = bag_size_generated.to(device);
        max_indices = max_indices.to(device);
        hashed_idx = hashed_idx.to(device);
    }

    auto expected_offsets2bag = make_offset2bag(offsets, indices);
    // generate expected output by python
    auto expected_hashed_index = torch::zeros({indices_num, num_feature}, torch::kLong);
    auto expected_output = torch::zeros({bag_num, num_feature});

    for (size_t i = 0; i < indices.size(0); ++i) {
        for (size_t j = 0; j < num_feature; ++j) {
            auto weight_idx = hash_func(indices[i].item().toLong(), j) % hashed_weights.size(0);
            expected_hashed_index[i][j] = weight_idx;
            expected_output[expected_offsets2bag[i].item().toLong()][j] += hashed_weights[weight_idx];
        }
    }

    // assert forward results are correct
    EXPECT_TRUE((expected_offsets2bag - offset2bag).abs().sum().item().toLong() == 0);
    EXPECT_TRUE(expected_hashed_index.equal(hashed_idx));
    EXPECT_TRUE(expected_output.equal(output));

    // the gradient of output, which is the input for backward.
    auto output_grad = torch::rand_like(expected_output);
    auto expected_weight_grad = torch::zeros_like(hashed_weights);

    // generate gradient for weight on CPU
    for (size_t i = 0; i < indices.size(0); ++i) {
        for (size_t j = 0; j < num_feature; ++j) {
            auto weight_idx = hash_func(indices[i].item().toLong(), j) % hashed_weights.size(0);
            expected_weight_grad[weight_idx] += output_grad[offset2bag[i].item().toLong()][j];
        }
    }


    // move all tensors to GPU
    {
        auto device = torch::kCUDA;
        hashed_weights = hashed_weights.to(device);
        indices = indices.to(device);
        offsets = offsets.to(device);
        offset2bag = offset2bag.to(device);
        bag_size = bag_size.to(device);
        max_indices = max_indices.to(device);
        hashed_idx = hashed_idx.to(device);
        output_grad = output_grad.to(device);
    }

    auto weight_grad = hashed_embedding_bag_backward(
            output_grad, indices, offsets, offset2bag, bag_size, max_indices, hashed_idx, hashed_weights.size(0), false,
            mode, num_feature);
    weight_grad = weight_grad.to(torch::kCPU);
    EXPECT_TRUE((weight_grad - expected_weight_grad).sum().item().toFloat() < 0.1);

}


TEST(hashed_embedding_bag_kernel, mean_mode) {
    int mode = 1;
    int bag_num = 18;
    int num_categories = 100;
    int num_feature = 200;
    int hashed_weight_size = 200;


    auto long_cuda_option = torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA);

    auto hashed_weights = torch::rand({hashed_weight_size}, torch::kCUDA);
    auto bag_size = torch::randint(0, 7, {bag_num}, long_cuda_option);
    int indices_num = bag_size.sum().item().toInt();
    auto indices = torch::randint(0, num_categories - 1, {indices_num}, long_cuda_option);
    auto offsets = torch::cat({
                                      torch::zeros(1, long_cuda_option),
                                      bag_size.cumsum(0).slice(0, 0, -1)});


    // run forward function on GPU
    auto [output, offset2bag, bag_size_generated, max_indices, hashed_idx] =
    hashed_embedding_bag_forward(hashed_weights, indices, offsets, mode, num_feature);

    {
        auto device = torch::kCPU;
        hashed_weights = hashed_weights.to(device);
        indices = indices.to(device);
        offsets = offsets.to(device);
        output = output.to(device);
        offset2bag = offset2bag.to(device);
        bag_size = bag_size.to(device);
        bag_size_generated = bag_size_generated.to(device);
        max_indices = max_indices.to(device);
        hashed_idx = hashed_idx.to(device);
    }

    auto expected_offsets2bag = make_offset2bag(offsets, indices);
    // generate expected output by python
    auto expected_hashed_index = torch::zeros({indices_num, num_feature}, torch::kLong);
    auto expected_output = torch::zeros({bag_num, num_feature});

    for (size_t i = 0; i < indices.size(0); ++i) {
        for (size_t j = 0; j < num_feature; ++j) {
            auto weight_idx = hash_func(indices[i].item().toLong(), j) % hashed_weights.size(0);
            expected_hashed_index[i][j] = weight_idx;
            expected_output[expected_offsets2bag[i].item().toLong()][j] += hashed_weights[weight_idx];
        }
    }


    expected_output /= bag_size.where(bag_size != 0, torch::ones_like(bag_size))
            .view({-1, 1}).expand({-1, expected_output.size(1)});

    // assert forward results are correct
    EXPECT_TRUE((expected_offsets2bag - offset2bag).abs().sum().item().toLong() == 0);
    EXPECT_TRUE(expected_hashed_index.equal(hashed_idx));
    EXPECT_TRUE(expected_output.equal(output));

    // the gradient of output, which is the input for backward.
    auto output_grad = torch::rand_like(expected_output);
    auto expected_weight_grad = torch::zeros_like(hashed_weights);

    // generate gradient for weight on CPU
    for (size_t i = 0; i < indices.size(0); ++i) {
        for (size_t j = 0; j < num_feature; ++j) {
            auto weight_idx = hash_func(indices[i].item().toLong(), j) % hashed_weights.size(0);
            expected_weight_grad[weight_idx] += output_grad[offset2bag[i].item().toLong()][j] / bag_size[offset2bag[i]];
        }
    }


    // move all tensors to GPU
    {
        auto device = torch::kCUDA;
        hashed_weights = hashed_weights.to(device);
        indices = indices.to(device);
        offsets = offsets.to(device);
        offset2bag = offset2bag.to(device);
        bag_size = bag_size.to(device);
        max_indices = max_indices.to(device);
        hashed_idx = hashed_idx.to(device);
        output_grad = output_grad.to(device);
    }

    auto weight_grad = hashed_embedding_bag_backward(
            output_grad, indices, offsets, offset2bag, bag_size, max_indices, hashed_idx, hashed_weights.size(0), false,
            mode, num_feature);
    weight_grad = weight_grad.to(torch::kCPU);
    EXPECT_TRUE((weight_grad - expected_weight_grad).sum().item().toFloat() < 0.1);

}

TEST(hashed_embedding_bag_kernel, max_mode) {
    int mode = 2;
    int bag_num = 18;
    int num_categories = 100;
    int num_feature = 200;
    int hashed_weight_size = 200;


    auto long_cuda_option = torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA);

    auto hashed_weights = torch::rand({hashed_weight_size}, torch::kCUDA);
    auto bag_size = torch::randint(0, 7, {bag_num}, long_cuda_option);
    int indices_num = bag_size.sum().item().toInt();
    auto indices = torch::randint(0, num_categories - 1, {indices_num}, long_cuda_option);
    auto offsets = torch::cat({
                                      torch::zeros(1, long_cuda_option),
                                      bag_size.cumsum(0).slice(0, 0, -1)});


    // run forward function on GPU
    auto [output, offset2bag, bag_size_generated, max_indices, hashed_idx] =
    hashed_embedding_bag_forward(hashed_weights, indices, offsets, mode, num_feature);

    {
        auto device = torch::kCPU;
        hashed_weights = hashed_weights.to(device);
        indices = indices.to(device);
        offsets = offsets.to(device);
        output = output.to(device);
        offset2bag = offset2bag.to(device);
        bag_size = bag_size.to(device);
        bag_size_generated = bag_size_generated.to(device);
        max_indices = max_indices.to(device);
        hashed_idx = hashed_idx.to(device);
    }

    auto expected_offsets2bag = make_offset2bag(offsets, indices);
    // generate expected output by python
    auto expected_hashed_index = torch::zeros({indices_num, num_feature}, torch::kLong);
    auto expected_output = torch::zeros({bag_num, num_feature});
    auto expected_max_indices = -torch::ones({bag_num, num_feature}, torch::kLong);

    for (size_t i = 0; i < indices.size(0); ++i) {
        for (size_t j = 0; j < num_feature; ++j) {
            auto weight_idx = hash_func(indices[i].item().toLong(), j) % hashed_weights.size(0);
            expected_hashed_index[i][j] = weight_idx;
            if (hashed_weights[weight_idx].item().toFloat() >
                expected_output[expected_offsets2bag[i].item().toLong()][j].item().toFloat())
            {
                expected_max_indices[expected_offsets2bag[i].item().toLong()][j] = weight_idx;
                expected_output[expected_offsets2bag[i].item().toLong()][j] = hashed_weights[weight_idx];
            }

        }
    }


    // assert forward results are correct
    EXPECT_TRUE((expected_offsets2bag - offset2bag).abs().sum().item().toLong() == 0);
    EXPECT_TRUE(expected_max_indices.equal(max_indices));
    EXPECT_TRUE(expected_hashed_index.equal(hashed_idx));
    EXPECT_TRUE(expected_output.equal(output));

    // the gradient of output, which is the input for backward.
    auto output_grad = torch::rand_like(expected_output);
    auto expected_weight_grad = torch::zeros_like(hashed_weights);

    // generate gradient for weight on CPU
    for (size_t i = 0; i < indices.size(0); ++i) {
        for (size_t j = 0; j < num_feature; ++j) {
            auto weight_idx = hash_func(indices[i].item().toLong(), j) % hashed_weights.size(0);
            if (expected_max_indices[offset2bag[i].item().toLong()][j].item().toLong() == weight_idx) {
                expected_weight_grad[weight_idx] += output_grad[offset2bag[i].item().toLong()][j];
            }
        }
    }


    // move all tensors to GPU
    {
        auto device = torch::kCUDA;
        hashed_weights = hashed_weights.to(device);
        indices = indices.to(device);
        offsets = offsets.to(device);
        offset2bag = offset2bag.to(device);
        bag_size = bag_size.to(device);
        max_indices = max_indices.to(device);
        hashed_idx = hashed_idx.to(device);
        output_grad = output_grad.to(device);
    }

    auto weight_grad = hashed_embedding_bag_backward(
            output_grad, indices, offsets, offset2bag, bag_size, max_indices, hashed_idx, hashed_weights.size(0), false,
            mode, num_feature);
    weight_grad = weight_grad.to(torch::kCPU);
    EXPECT_TRUE((weight_grad - expected_weight_grad).sum().item().toFloat() < 0.1);

}

int main() {
    testing::InitGoogleTest();
    RUN_ALL_TESTS();
    return 0;
}

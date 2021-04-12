import torch
import HashedEmbeddingBag
import hashed_embedding_bag


def make_offset2bag(offsets, indices):
    offsets2bag = torch.zeros(indices.size(0) + 1, dtype=indices.dtype, device=offsets.device)
    offsets2bag.index_add_(0, offsets, torch.ones_like(offsets, memory_format=torch.legacy_contiguous_format))
    offsets2bag[0] -= 1
    offsets2bag = offsets2bag.cumsum(0)
    offsets2bag.resize_(indices.size(0))
    return offsets2bag


def test_hashedEmbeddingBag():
    # the 'sum' mode
    mode = 0

    bag_num = 18

    num_categories = 100
    num_feature = 200

    hashed_weight_size = 200

    # generate random weight and input for testing
    hashed_weights = torch.rand(hashed_weight_size)
    bag_size = torch.randint(low=0, high=7, size=(bag_num,))
    indices_num = bag_size.sum().item()

    indices = torch.randint(low=0, high=num_categories - 1, size=(indices_num,))
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), bag_size.cumsum(dim=0)[:-1]])

    # move all inputs to GPU
    device = torch.cuda.current_device()
    hashed_weights = hashed_weights.to(device)
    indices = indices.to(device)
    offsets = offsets.to(device)

    # run forward function on GPU
    output, offset2bag, bag_size, max_indices, hashed_idx = \
        hashed_embedding_bag.forward(hashed_weights, indices, offsets, mode, num_feature)

    # move weight, inputs, and outputs to CPU
    device = torch.device("cpu")
    hashed_weights = hashed_weights.to(device)
    indices = indices.to(device)
    offsets = offsets.to(device)
    output = output.to(device)
    offset2bag = offset2bag.to(device)
    bag_size = bag_size.to(device)
    max_indices = max_indices.to(device)
    hashed_idx = hashed_idx.to(device)

    expected_offsets2bag = make_offset2bag(offsets, indices)
    # generate expected output by python
    expected_hashed_index = torch.zeros((indices_num, num_feature), dtype=torch.long)
    expected_output = torch.zeros(bag_num, num_feature)
    for i in range(indices.size(0)):
        for j in range(num_feature):
            weight_idx = hashed_embedding_bag.hash(indices[i].item(), j) % hashed_weights.size(0)
            expected_hashed_index[i, j] = weight_idx
            expected_output[expected_offsets2bag[i].item(), j] += hashed_weights[weight_idx]

    # assert forward results are correct
    assert ((expected_offsets2bag - offset2bag).abs().sum().item() == 0)
    assert (expected_hashed_index.equal(hashed_idx))
    assert (expected_output.equal(output))

    # the gradient of output, which is the input for backward.
    output_grad = torch.rand_like(expected_output)

    expected_weight_grad = torch.zeros_like(hashed_weights)

    # generate gradient for weight in python
    for i in range(indices.size(0)):
        for j in range(num_feature):
            weight_idx = hashed_embedding_bag.hash(indices[i].item(), j) % hashed_weights.size(0)
            expected_weight_grad[weight_idx] += output_grad[offset2bag[i].item(), j]

    # move all tensors to GPU
    device = torch.cuda.current_device()
    hashed_weights = hashed_weights.to(device)
    indices = indices.to(device)
    offsets = offsets.to(device)
    offset2bag = offset2bag.to(device)
    bag_size = bag_size.to(device)
    max_indices = max_indices.to(device)
    hashed_idx = hashed_idx.to(device)

    output_grad = output_grad.to(device)
    weight_grad = hashed_embedding_bag.backward(
        output_grad, indices, offsets, offset2bag, bag_size, max_indices, hashed_idx, hashed_weights.size(0), False,
        mode, num_feature)
    weight_grad = weight_grad.cpu()

    assert ((weight_grad - expected_weight_grad).sum().item() < 0.1)

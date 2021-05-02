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

def test_hashedEmbeddingBag_single():
    bag_num = 180

    num_categories = 100
    num_feature = 200

    hashed_weight_size = 200

    # generate random weight and input for testing
    hashed_weights = torch.rand(hashed_weight_size, requires_grad=True)

    embedding = HashedEmbeddingBag.HashedEmbeddingBag(
        num_categories, num_feature, compression=0.1, _weight=hashed_weights)
    embedding = embedding.cuda()

    indices_num = bag_num

    indices = torch.randint(low=0, high=num_categories - 1, size=(indices_num, 1))

    # move all inputs to GPU
    device = torch.cuda.current_device()
    indices = indices.to(device)

    output = embedding.forward(indices)

    # give a 'weight' to different locations in output, so that the element in output_grad is different from each other.
    x = torch.rand_like(output).cuda()
    loss = (output * x).sum()
    loss.backward()

    # move weight, inputs, and outputs to CPU
    device = torch.device("cpu")
    hashed_weights = hashed_weights.to(device)
    indices = indices.to(device)
    output = output.to(device)

    # generate expected output by python
    expected_hashed_index = torch.zeros((indices_num, num_feature), dtype=torch.long)
    expected_output = torch.zeros(bag_num, num_feature)
    for i in range(indices.size(0)):
        for j in range(num_feature):
            weight_idx = hashed_embedding_bag.hash(indices[i].item(), j) % hashed_weights.size(0)
            expected_hashed_index[i, j] = weight_idx
            expected_output[i, j] += hashed_weights[weight_idx]

    # assert forward results are correct
    assert (expected_output.equal(output))

    # the gradient of output, which is the input for backward.
    output_grad = x.cpu()

    expected_weight_grad = torch.zeros_like(hashed_weights)

    # generate gradient for weight in python
    for i in range(indices.size(0)):
        for j in range(num_feature):
            weight_idx = hashed_embedding_bag.hash(indices[i].item(), j) % hashed_weights.size(0)
            expected_weight_grad[weight_idx] += output_grad[i, j]

    # move all tensors to GPU
    device = torch.cuda.current_device()
    hashed_weights = hashed_weights.to(device)
    indices = indices.to(device)

    assert ((embedding.hashed_weight.grad.data.cpu() - expected_weight_grad).sum().item() < 0.1)


def test_HashedEmbeddingBagAPI_mean():
    bag_num = 18

    num_categories = 10
    num_feature = 5

    hashed_weight_size = 200

    # generate random weight and input for testing
    hashed_weights = torch.rand(hashed_weight_size)

    embedding = HashedEmbeddingBag.HashedEmbeddingBag(
        num_categories, num_feature, compression=0.1, mode="mean", _weight=hashed_weights)
    embedding = embedding.cuda()

    bag_size = torch.randint(low=0, high=3, size=(bag_num,))
    indices_num = bag_size.sum().item()

    indices = torch.randint(low=0, high=num_categories - 1, size=(indices_num,))
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), bag_size.cumsum(dim=0)[:-1]])

    # move all inputs to GPU
    device = torch.cuda.current_device()
    indices = indices.to(device)
    offsets = offsets.to(device)

    x = embedding.forward(indices, offsets)
    loss = x.sum()
    loss.backward()


def test_HashedEmbeddingBagAPI_max():
    bag_num = 18

    num_categories = 10
    num_feature = 5

    hashed_weight_size = 200

    # generate random weight and input for testing
    hashed_weights = torch.rand(hashed_weight_size)

    embedding = HashedEmbeddingBag.HashedEmbeddingBag(num_categories, num_feature, compression=0.1, mode="max")
    embedding = embedding.cuda()

    bag_size = torch.randint(low=0, high=3, size=(bag_num,))
    indices_num = bag_size.sum().item()

    indices = torch.randint(low=0, high=num_categories - 1, size=(indices_num,))
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), bag_size.cumsum(dim=0)[:-1]])

    # move all inputs to GPU
    device = torch.cuda.current_device()
    indices = indices.to(device)
    offsets = offsets.to(device)

    x = embedding.forward(indices, offsets)
    loss = x.sum()
    loss.backward()


def test_HashedEmbeddingBagAPI_sum():
    bag_num = 18

    num_categories = 10
    num_feature = 5

    hashed_weight_size = 200

    # generate random weight and input for testing
    hashed_weights = torch.rand(hashed_weight_size)

    embedding = HashedEmbeddingBag.HashedEmbeddingBag(num_categories, num_feature, compression=0.1, mode="sum")
    embedding = embedding.cuda()

    bag_size = torch.randint(low=0, high=3, size=(bag_num,))
    indices_num = bag_size.sum().item()

    indices = torch.randint(low=0, high=num_categories - 1, size=(indices_num,))
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), bag_size.cumsum(dim=0)[:-1]])

    # move all inputs to GPU
    device = torch.cuda.current_device()
    indices = indices.to(device)
    offsets = offsets.to(device)

    x = embedding.forward(indices, offsets)
    loss = x.sum()
    loss.backward()

def test_HashedEmbeddingBagAPI_single():
    bag_num = 200

    num_categories = 10
    num_feature = 5

    hashed_weight_size = 200

    # generate random weight and input for testing
    hashed_weights = torch.rand(hashed_weight_size)

    embedding = HashedEmbeddingBag.HashedEmbeddingBag(num_categories, num_feature, compression=0.1, mode="sum")
    embedding = embedding.cuda()

    indices_num = bag_num

    indices = torch.randint(low=0, high=num_categories - 1, size=(indices_num, 1))

    # move all inputs to GPU
    device = torch.cuda.current_device()
    indices = indices.to(device)

    x = embedding.forward(indices)
    loss = x.sum()
    loss.backward()

def test_HashedEmbeddingBagAPI_embeddding():
    bag_num = 200

    num_categories = 10
    num_feature = 5

    hashed_weight_size = 200

    # generate random weight and input for testing
    hashed_weights = torch.rand(hashed_weight_size)

    embedding = HashedEmbeddingBag.HashedEmbedding(num_categories, num_feature, compression=0.1)
    embedding = embedding.cuda()

    indices_num = bag_num

    indices = torch.randint(low=0, high=num_categories - 1, size=(indices_num, ))

    # move all inputs to GPU
    device = torch.cuda.current_device()
    indices = indices.to(device)

    x = embedding.forward(indices)
    loss = x.sum()
    loss.backward()
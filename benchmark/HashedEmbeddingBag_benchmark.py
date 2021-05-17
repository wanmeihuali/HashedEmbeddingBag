import torch
import HashedEmbeddingBag
import time
import statistics

def benchmark():
    bag_num = 4096

    num_categories = 300000
    num_feature = 16

    compression = 0.001
    mode = "mean"
    iterations = 10000
    experiments = 10


    embedding = HashedEmbeddingBag.HashedEmbeddingBag(
        num_categories, num_feature, compression=compression, mode=mode)
    embedding = embedding.cuda()

    original_embedding = torch.nn.EmbeddingBag(
        num_categories, num_feature, mode=mode)
    original_embedding = original_embedding.cuda()

    bag_size = torch.randint(low=0, high=3, size=(bag_num,))
    indices_num = bag_size.sum().item()

    indices = torch.randint(low=0, high=num_categories - 1, size=(indices_num,))
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), bag_size.cumsum(dim=0)[:-1]])

    # move all inputs to GPU
    device = torch.cuda.current_device()
    indices = indices.to(device)
    offsets = offsets.to(device)

    print(f"Settings: bag_num = {bag_num}, dim size = {num_feature}, mode = {mode}")

    durations = []
    for exp_i in range(experiments):
        start_time = time.time()
        for _ in range(iterations):
            _ = embedding.forward(indices, offsets)
            torch.cuda.synchronize()
        end_time = time.time()
        duration = end_time - start_time
        durations.append(duration)
    print(f"Hashed Embedding average running time = {sum(durations) * 1000000 / iterations / experiments}us, stdev time = {statistics.stdev(durations)}, compression = {compression}")


    durations = []
    for exp_i in range(experiments):
        start_time = time.time()
        for _ in range(iterations):
            _ = original_embedding.forward(indices, offsets)
            torch.cuda.synchronize()
        end_time = time.time()
        duration = end_time - start_time
        durations.append(duration)
    print(f"Original Embedding average running time = {sum(durations) * 1000000 / iterations / experiments}us, stdev time = {statistics.stdev(durations)}")



if __name__ == "__main__":
    benchmark()

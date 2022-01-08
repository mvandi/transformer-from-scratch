import time

import torch

from transformer import Transformer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    batch_size = 2
    src_vocab_size = 4
    tgt_vocab_size = 4

    src = torch.tensor([
        [1, 2, 3, 0, 0, 0],
        [1, 2, 3, 1, 0, 0]
    ]).to(device)
    tgt = torch.tensor([
        [1, 2, 3, 0, 0, 0],
        [1, 2, 3, 1, 0, 0]
    ]).to(device)

    model = Transformer(
        source_vocab_size=src_vocab_size,
        target_vocab_size=tgt_vocab_size,
        source_padding_idx=0,
        target_padding_idx=0,
    ).to(device)
    model.eval()

    for _ in range(2):
        start = time.perf_counter()
        out = model(src, tgt[:, :-1])
        elapsed = time.perf_counter() - start
        print(f'{out.size()} time: {elapsed}')

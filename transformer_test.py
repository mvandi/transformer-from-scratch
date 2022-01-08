import time
import torch

from transformer import Transformer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    batch_size = 3
    src_vocab_size = 100
    tgt_vocab_size = 10

    src = torch.randint(src_vocab_size, size=(batch_size, 30)).to(device)
    tgt = torch.randint(tgt_vocab_size, size=(batch_size, 30)).to(device)

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_padding_idx=0,
        tgt_padding_idx=0,
    ).to(device)
    model.eval()
    for _ in range(2):
        start = time.perf_counter()
        out = model(src, tgt[:, :-1])
        elapsed = time.perf_counter() - start
        print(f'{out.size()} time: {elapsed}')

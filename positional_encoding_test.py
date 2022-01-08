from matplotlib import pyplot as plt

import functional as F

if __name__ == '__main__':
    seq_len = 2048
    d_model = 512
    pos_enc = F.positional_encoding_matrix(seq_len, d_model, dtype=torch.float32)
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(pos_enc, vmin=-1, vmax=1)
    plt.xlabel('Word embedding')
    plt.xlim((0, d_model))
    plt.ylabel('Position')
    plt.ylim((seq_len, 0))
    plt.colorbar()
    plt.show()

from axonn import axonn as ax
from axonn.intra_layer import Embedding
import torch


if __name__ == "__main__":
    ax.init(G_intra_r=4)
    V = 50304
    H = 1024
    B = 4
    S = 2048
    layer = Embedding(V, H, expert_mode=True).cuda().half()
    x = torch.randint(low=0, high=V, size=[B, S], device='cuda')
    y = layer(x)
    print(y.shape)

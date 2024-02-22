import torch.nn as nn
import torch

class FC_Net(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, output_size):
        super(FC_Net, self).__init__()
        self.embed = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList([FC_Net_Layer(hidden_size) for _ in range(num_layers)])
        self.clf = nn.Linear(hidden_size, output_size)
        self.checkpoint_activations = False

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.embed(x)
        for layer in self.layers:
            if not self.checkpoint_activations:
                x = layer(x)
            else:
                x = torch.utils.checkpoint.checkpoint(layer, x)
        x = self.clf(x)
        return x

class FC_Net_Layer(nn.Module):
    def __init__(self, hidden_size):
        super(FC_Net_Layer, self).__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear_1 = nn.Linear(in_features=hidden_size, out_features=4 * hidden_size)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(in_features = 4 * hidden_size, out_features = hidden_size)

    def forward(self, x):
        h = self.norm(x)
        h = self.linear_1(h)
        h = self.relu(h)
        h = self.linear_2(h)
        return h + x

if __name__ == "main":
    net = FC_Net(num_layers=2, input_size=256, hidden_size=1024, output_size=10).cuda()
    x = torch.rand(64, 256).cuda()
    y = net(x)
    print(y.shape, y.device)

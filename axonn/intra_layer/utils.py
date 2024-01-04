def divide(a, b):
    assert a % b == 0
    return a // b

@torch.no_grad()
def default_init_method(weight):
    return torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
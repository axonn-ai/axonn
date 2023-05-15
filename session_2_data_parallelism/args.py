import argparse

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--image-size", default=64, type=int)
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-activations", action='store_true', default=False)
    return parser

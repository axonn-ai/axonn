import argparse

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--image-size", default=64, type=int)
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--G-intra-r", type=int, help="degree of intra-layer parallelism (row)")
    parser.add_argument("--G-intra-c", type=int, help="degree of intra-layer parallelism (column)")
    parser.add_argument("--G-data", type=int, help="degree of data parallelism")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-activations", action='store_true', default=False)
    return parser

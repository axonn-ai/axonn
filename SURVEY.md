## To run on VGG-16

### This is an example to run vgg16 on 2 nodes (6 GPUs per node) on Summit
jsrun --smpiargs='-gpu' -n 2 -a 6 -c 42 -g 6 bash run_vgg.sh

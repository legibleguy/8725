#!/bin/bash

# Create a new detached session
tmux new-session -d -s resnet_cifar

# Create a new window and run the script with a different set of parameters
tmux new-window -t resnet_cifar
tmux send-keys -t resnet_cifar:0 "python resnet_cifar.py --lr 0.001 --momentum 0.8 --batch_size 12 --epochs 128 > log2.txt" C-m

# ... repeat for as many windows as you need ...

# Finally, attach to the session
tmux attach -t my_session

import pdb
import argparse
import time


import torch
import torch.nn as nn
from torch.optim import Adam


from models.definitions.GAT import GAT
from utils.data_loading import load_graph_data
from utils.constants import *
import utils.utils as utils


def train_gat_cora(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!

    # Step 1: load the graph data
    node_features, node_labels, edge_index, train_indices, val_indices, test_indices = load_graph_data(config, device)
    ### BUG: node_features vary in AR
    # What is edg-index? it is a representation of the edges of the graph
    #
    graph_data = (node_features, edge_index)



    # Step 2: prepare the model
    gat = GAT(
        num_of_layers=config['num_of_layers'],
        num_heads_per_layer=config['num_heads_per_layer'],
        num_features_per_layer=config['num_features_per_layer'],
        add_skip_connection=config['add_skip_connection'],
        bias=config['bias'],
        dropout=config['dropout'],
        layer_type=config['layer_type'],
        log_attention_weights=False  # no need to store attentions, used only in playground.py for visualizations
    ).to(device)

    # Step 3: Prepare other training related utilities (loss & optimizer and decorator function)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = Adam(gat.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    if phase == LoopPhase.TRAIN:
        gat.train()
    else:
        gat.eval()

    # Do a forwards pass and extract only the relevant node scores (train/val or test ones)
    # Note: [0] just extracts the node_features part of the data (index 1 contains the edge_index)
    # shape = (N, C) where N is the number of nodes in the split (train/val/test) and C is the number of classes
    nodes_unnormalized_scores = gat(graph_data)[0].index_select(node_dim, node_indices)

    # Example: let's take an output for a single node on Cora - it's a vector of size 7 and it contains unnormalized
    # scores like: V = [-1.393,  3.0765, -2.4445,  9.6219,  2.1658, -5.5243, -4.6247]
    # What PyTorch's cross entropy loss does is for every such vector it first applies a softmax, and so we'll
    # have the V transformed into: [1.6421e-05, 1.4338e-03, 5.7378e-06, 0.99797, 5.7673e-04, 2.6376e-07, 6.4848e-07]
    # secondly, whatever the correct class is (say it's 3), it will then take the element at position 3,
    # 0.99797 in this case, and the loss will be -log(0.99797). It does this for every node and applies a mean.
    # You can see that as the probability of the correct class for most nodes approaches 1 we get to 0 loss! <3
    loss = cross_entropy_loss(nodes_unnormalized_scores, gt_node_labels)


    if phase == LoopPhase.TRAIN:
        optimizer.zero_grad()  # clean the trainable weights gradients in the computational graph (.grad fields)
        loss.backward()  # compute the gradients for every trainable weight in the computational graph
        optimizer.step()  # apply the gradients to weights



if __name__ == '__main__':

    # Train the graph attention network (GAT)
    train_gat_cora(get_training_args())

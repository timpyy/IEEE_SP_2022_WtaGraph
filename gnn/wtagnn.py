import torch as th
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_feats_node, in_feats_edge, out_feats, num_heads=1, activation=None, dropout=0.0, bias=True):
        super(GATLayer, self).__init__()
        self.in_feats_node = in_feats_node
        self.in_feats_edge = in_feats_edge
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout else None

        # Linear transformations for multi-head attention
        self.fc_node = nn.Linear(in_feats_node, out_feats * num_heads, bias=False)
        self.fc_edge = nn.Linear(in_feats_edge, out_feats * num_heads, bias=False)

        # Attention weights
        self.attn_l = nn.Parameter(th.FloatTensor(size=(num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(num_heads, out_feats)))
        self.attn_e = nn.Parameter(th.FloatTensor(size=(num_heads, out_feats)))

        # Learnable neighbor attention matrix
        self.neighbor_attention_matrix = nn.Parameter(th.Tensor(out_feats * num_heads, in_feats_edge))
        nn.init.xavier_uniform_(self.neighbor_attention_matrix)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc_node.weight)
        nn.init.xavier_normal_(self.fc_edge.weight)
        nn.init.xavier_normal_(self.attn_l)
        nn.init.xavier_normal_(self.attn_r)
        nn.init.xavier_normal_(self.attn_e)
        nn.init.xavier_uniform_(self.neighbor_attention_matrix)

    def edge_attention(self, edges, neighbor_edges):
        """
        Computes unnormalized attention scores based on nodes, edge, and neighbor edge features.
        """
        el = (edges.src['z'] * self.attn_l).sum(dim=-1)  # (E, H)
        er = (edges.dst['z'] * self.attn_r).sum(dim=-1)  # (E, H)

        transformed_z_e = edges.data['z_e'].view(-1, self.out_feats * self.num_heads)
        weighted_z_e = transformed_z_e @ self.neighbor_attention_matrix.T
        weighted_z_e = weighted_z_e.view(-1, self.num_heads, self.out_feats)
        ee = (weighted_z_e * self.attn_e).sum(dim=-1)  # (E, H)

        # Dynamically retrieve neighbor features
        neighbor_contribution = neighbor_edges[edges.data[dgl.EID]].to(g.device)
        neighbor_weights = self.neighbor_attention_matrix @ neighbor_contribution.T
        neighbor_weights = neighbor_weights.T.view(-1, self.num_heads, self.out_feats)
        neighbor_contribution = (neighbor_weights * self.attn_e).sum(dim=-1)  # (E, H)

        e = F.leaky_relu(el + er + ee + neighbor_contribution)  # Combine all contributions
        return {'e': e}

    def forward(self, g, nf, ef, neighbor_edges):
        # Move neighbor_edges to the graph's device temporarily
        neighbor_edges = neighbor_edges.to(g.device)

        ef = ef.view(-1, self.in_feats_edge)
        z = self.fc_node(nf).view(-1, self.num_heads, self.out_feats)
        z_e = self.fc_edge(ef).view(-1, self.num_heads, self.out_feats)

        g.ndata['z'] = z
        g.edata['z_e'] = z_e

        # Dynamically pass neighbor_edges to edge_attention
        g.apply_edges(lambda edges: self.edge_attention(edges, neighbor_edges))

        g.update_all(self.message_func, self.reduce_func)

        n_out = g.ndata.pop('h').view(-1, self.num_heads * self.out_feats)
        e_out = g.edata.pop('z_e').view(-1, self.num_heads * self.out_feats)
        return n_out, e_out

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e'], 'z_e': edges.data['z_e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = th.sum(alpha.unsqueeze(-1) * nodes.mailbox['z'], dim=1)
        return {'h': h}



class WTAGNN(nn.Module):
    def __init__(self, g, input_node_feat_size, input_edge_feat_size, n_hidden, n_classes,
                 n_layers, n_heads, activation, dropout):
        super(WTAGNN, self).__init__()
        # Input layer
        self.layers = nn.ModuleList()
        self.layers.append(GATLayer(input_node_feat_size, input_edge_feat_size, n_hidden, n_heads, activation, dropout))

        # Hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(GATLayer(n_hidden * n_heads, n_hidden * n_heads, n_hidden, n_heads, activation, dropout))

        # Output layers
        self.node_out_layer = GATLayer(n_hidden * n_heads, n_hidden * n_heads, n_classes, num_heads=1, activation=None,
                                       dropout=dropout)
        self.edge_transform_layer = nn.Linear(n_classes, n_hidden * n_heads)  # Transform layer for edge features
        self.edge_out_layer = nn.Linear(n_hidden * n_heads, n_classes)

    def forward(self, g, nf, ef, neighbor_edges):
        for layer in self.layers:
            nf, ef = layer(g, nf, ef, neighbor_edges)

        # Compute node logits and updated edge features
        n_logits, ef = self.node_out_layer(g, nf, ef, neighbor_edges)

        # Transform edge features to match edge_out_layer input requirements
        ef = self.edge_transform_layer(ef.view(-1, ef.size(-1)))

        # Compute edge logits
        e_logits = self.edge_out_layer(ef)

        return n_logits, e_logits


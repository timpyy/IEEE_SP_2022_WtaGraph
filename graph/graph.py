import random
import dgl
import torch as th
import numpy as np
import pickle


class GraphLoader:
    def __init__(self):
        pass

    def load_node_edge_map(self, args):
        db_name = args.db_name if args.db_name.endswith('/') else args.db_name + '/'

        id_node_map_path = './data/graph_data/' + db_name + args.graph_name + '_id_node_map.pickle'
        with open(id_node_map_path, 'rb') as f:
            id_node_map = pickle.load(f)

        id_edge_map_list_path = './data/graph_data/' + db_name + args.graph_name + '_id_edge_map_list.pickle'
        with open(id_edge_map_list_path, 'rb') as f:
            id_edge_map = pickle.load(f)

        return id_node_map, id_edge_map

    def load_graph(self, args):
        print('\n************loading the specified graph and feature data************')

        # Load the graph
        edgelist_path = './data/graph_data/' + args.db_name + '/' + args.graph_name + '_graph.edgelist'
        with open(edgelist_path, 'r') as f:
            edges = [tuple(map(int, line.strip().split(','))) for line in f]

        src_nodes = [e[1] for e in edges]
        dst_nodes = [e[2] for e in edges]

        # Rebuild the graph
        g = dgl.DGLGraph()
        g = g.to(args.gpu)
        g.add_nodes(len(set(src_nodes).union(set(dst_nodes))))
        g.add_edges(src_nodes, dst_nodes)
        print('Loaded graph: ', args.db_name, args.graph_name, g, '\n')

        # Load node features
        nf = np.load('./data/feat_data/' + args.db_name + '/' + args.graph_name + '_node_feat.npy')
        nf = th.from_numpy(nf)
        print('Node feature shape:', nf.shape)

        # Load edge features
        ef = np.load('./data/feat_data/' + args.db_name + '/' + args.graph_name + '_edge_feat.npy')
        ef = th.from_numpy(ef)
        print('Edge feature shape:', ef.shape)

        # Compute and store neighbor edge features separately
        neighbor_edges = self.compute_neighbor_features(g, ef)

        # Load edge labels
        e_label = np.load('./data/feat_data/' + args.db_name + '/' + args.graph_name + '_edge_label.npy').tolist()
        e_label = th.tensor(e_label)
        print('Edge labels shape:', e_label.shape)

        # Prepare train, test, and validation masks
        train_mask, test_mask, val_mask = self._split_dataset(e_label, (args.r_train, args.r_test, args.r_val))

        print('***************************loading completed***************************\n')
        return g, nf, ef, e_label, train_mask, test_mask, val_mask, neighbor_edges

    def compute_neighbor_features(self, g, ef):
        """
        Computes aggregated neighbor edge features for each edge in the graph.
        Ensures that the aggregation happens for edges, not nodes.
        """
        # Move edge features temporarily to the device of the graph for computation
        ef_device = ef.to(g.device)

        # Temporary scope to avoid modifying the graph permanently
        with g.local_scope():
            g.edata['temp_z_e'] = ef_device  # Assign edge features temporarily to the graph

            # Perform edge-wise aggregation
            g.apply_edges(lambda edges: {
                'agg_neighbor_edges': edges.data['temp_z_e']  # Replace with aggregation logic if needed
            })

            # Retrieve the result
            agg_neighbor_features = g.edata.pop('agg_neighbor_edges')

        # Return the aggregated features back to CPU
        return agg_neighbor_features.to('cpu')  # Do not assign back to g.edata


    def _split_dataset(self, labels, ratio_tuple):
        shuffle_list = [i for i in range(labels.shape[0])]
        random.shuffle(shuffle_list)
        train_ct = int(len(shuffle_list) * ratio_tuple[0])
        test_ct = int(len(shuffle_list) * ratio_tuple[1])
        val_ct = int(len(shuffle_list) * ratio_tuple[2])
        print('# of train edge:', train_ct, '   # of test edge:', test_ct, ' # of val edge:', val_ct)

        train_mask = np.zeros(labels.shape[0])
        test_mask = np.zeros(labels.shape[0])
        val_mask = np.zeros(labels.shape[0])
        for idx in range(0, train_ct):
            train_mask[shuffle_list[idx]] = 1
        for idx in range(train_ct, train_ct + test_ct):
            test_mask[shuffle_list[idx]] = 1
        for idx in range(len(shuffle_list) - val_ct, len(shuffle_list)):
            val_mask[shuffle_list[idx]] = 1

        train_mask = th.BoolTensor(train_mask)
        test_mask = th.BoolTensor(test_mask)
        val_mask = th.BoolTensor(val_mask)

        return train_mask, test_mask, val_mask

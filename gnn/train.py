import time
import numpy as np
import torch as th
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

from gnn.eval import performance, evaluate
from gnn.wtagnn import WTAGNN  # Ensure this imports your modified WTAGNN
from graph.graph import GraphLoader

def start_train(args):
    th.autograd.set_detect_anomaly(True)
    gloader = GraphLoader()
    g, nf, ef, e_label, edge_train_mask, edge_test_mask, edge_val_mask = gloader.load_graph(args)

    # Ensure GPU utilization by moving graph and data to GPU
    device = th.device(f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu')
    g = g.to(device)
    nf, ef, e_label = nf.to(device), ef.to(device), e_label.to(device)
    edge_train_mask, edge_val_mask, edge_test_mask = (
        edge_train_mask.to(device),
        edge_val_mask.to(device),
        edge_test_mask.to(device),
    )

    # Node mask creation (optional, update as necessary)
    num_nodes = g.num_nodes()
    node_train_mask = th.ones(num_nodes, dtype=th.bool, device=device)  # Example: All nodes part of training
    node_val_mask = th.ones(num_nodes, dtype=th.bool, device=device)
    node_test_mask = th.ones(num_nodes, dtype=th.bool, device=device)

    print('Node feature size:', nf.shape)
    print('Edge feature size:', ef.shape)

    n_classes = 2
    input_node_feat_size = nf.shape[1]
    input_edge_feat_size = ef.shape[1]

    print('\n************initialize model************')
    # Initialize model and move to GPU
    model = WTAGNN(g, input_node_feat_size, input_edge_feat_size,
                   args.n_hidden, n_classes, args.n_layers, args.n_heads, F.relu, args.dropout).to(device)
    print(model)

    # Loss function and optimizer
    loss_fcn = th.nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print('\n************start training************')
    dur, max_acc = [], -1
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()

        # Forward pass with node and edge features
        n_logits, e_logits = model(g, nf, ef)

        # Compute loss for nodes and edges
        node_loss = loss_fcn(n_logits[node_train_mask], th.zeros_like(n_logits[node_train_mask][:, 0].long()))
        edge_loss = loss_fcn(e_logits[edge_train_mask], e_label[edge_train_mask])
        loss = node_loss + edge_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        # Safely calculate average duration
        avg_dur = np.mean(dur) if dur else 0.0

        # Evaluate on validation set
        acc, predictions, labels = evaluate(model, g, nf, ef, e_label, edge_val_mask)

        # Save best model
        if acc > max_acc:
            max_acc = acc
            th.save(model.state_dict(), f'./output/best.model.{args.model_name}')

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(epoch, avg_dur, loss.item(),
                                            acc, g.number_of_edges() / avg_dur / 1000 if avg_dur > 0 else 0))

    # Load and test the best model
    best_model = WTAGNN(g, input_node_feat_size, input_edge_feat_size,
                        args.n_hidden, n_classes, args.n_layers, args.n_heads, F.relu, args.dropout).to(device)
    best_model.load_state_dict(th.load(f'./output/best.model.{args.model_name}'))

    acc, predictions, labels = evaluate(best_model, g, nf, ef, e_label, edge_test_mask)
    precision, recall, tnr, tpr, f1 = performance(predictions.tolist(), labels.tolist(), acc)


def start_train_cv(args):
    gloader = GraphLoader()
    g, nf, ef, e_label, _, _, _ = gloader.load_graph(args)

    device = th.device(f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu')
    g = g.to(device)
    nf, ef, e_label = nf.to(device), ef.to(device), e_label.to(device)

    n_classes = 2
    input_node_feat_size, input_edge_feat_size = nf.shape[1], ef.shape[1]

    print('\n************start training for {:d} folds************'.format(args.fold))
    kf = StratifiedKFold(n_splits=args.fold, shuffle=True)
    kf.get_n_splits()

    fold = 0
    total_precision = total_acc = total_recall = 0
    for train_index, test_index in kf.split(e_label, e_label):
        fold += 1
        print('\nFold #: ', fold)
        train_mask = th.BoolTensor((np.arange(len(e_label))[:, None] == train_index).sum(axis=1)).to(device)
        test_mask = th.BoolTensor((np.arange(len(e_label))[:, None] == test_index).sum(axis=1)).to(device)

        model = WTAGNN(g, input_node_feat_size, input_edge_feat_size,
                       args.n_hidden, n_classes, args.n_layers, args.n_heads, F.relu, args.dropout).to(device)

        loss_fcn = th.nn.CrossEntropyLoss()
        optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        dur, max_acc = [], -1
        for epoch in range(args.n_epochs):
            model.train()
            if epoch >= 3:
                t0 = time.time()

            n_logits, e_logits = model(g, nf, ef)
            node_loss = loss_fcn(n_logits[train_mask], e_label[train_mask])
            edge_loss = loss_fcn(e_logits[train_mask], e_label[train_mask])
            loss = node_loss + edge_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            acc, predictions, labels = evaluate(model, g, nf, ef, e_label, test_mask)
            if acc > max_acc:
                max_acc = acc
                th.save(model.state_dict(), f'./output/best.model.{args.model_name}.fold.{fold}')

            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                  "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur) if dur else 0.0, loss.item(),
                                                acc, g.number_of_edges() / (np.mean(dur) if dur else 1) / 1000))

        best_model = WTAGNN(g, input_node_feat_size, input_edge_feat_size,
                            args.n_hidden, n_classes, args.n_layers, args.n_heads, F.relu, args.dropout).to(device)
        best_model.load_state_dict(th.load(f'./output/best.model.{args.model_name}.fold.{fold}'))

        acc, predictions, labels = evaluate(best_model, g, nf, ef, e_label, test_mask)
        precision, recall, tnr, tpr, f1 = performance(predictions.tolist(), labels.tolist(), acc)

        total_precision += precision
        total_acc += acc
        total_recall += recall

    print('\n************training done! Averaged model performance************')
    print('Acc/Precision/Recall: ',
          f"{total_acc / args.fold * 100:.2f}% / {total_precision / args.fold * 100:.2f}% / {total_recall / args.fold * 100:.2f}%")

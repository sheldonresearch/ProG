pre_train
============

Pre-train your GNN model
--------------------

The following codes present a simple example on how to pre-train a GNN model via GraphCL. You can also find a integrated
function ``pretrain()`` in ``no_meta_demo.py``.

.. code-block:: python

    from ProG.utils import mkdir, load_data4pretrain
    from ProG import PreTrain

    mkdir('./pre_trained_gnn/')

    pretext = 'GraphCL'  # 'GraphCL', 'SimGRACE'
    gnn_type = 'TransformerConv'  # 'GAT', 'GCN'
    dataname, num_parts, batch_size = 'CiteSeer', 200, 10

    print("load data...")
    graph_list, input_dim, hid_dim = load_data4pretrain(dataname, num_parts)

    print("create PreTrain instance...")
    pt = PreTrain(pretext, gnn_type, input_dim, hid_dim, gln=2)

    print("pre-training...")
    pt.train(dataname, graph_list, batch_size=batch_size,
            aug1='dropN', aug2="permE", aug_ratio=None,
            lr=0.01, decay=0.0001, epochs=100)

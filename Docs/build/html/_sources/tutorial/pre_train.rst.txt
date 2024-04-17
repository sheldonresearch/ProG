pre_train
============

- We summarized all the possible ways of pre-training in **academic research**
include:
::
    - Edge Prediction
    - GraphCL
    - SimGRACE
    - and even more

To pre Train your model you basicly need those steps

- **first**: determine which model you will use, what's the hidden dimension and number of hidden layers
.. code-block:: python

    gln = number of hidden layers
    hid_dim = hidden dimension
    gnn_type = model you what use

- **second**: determine the dataset and how many shots you what use
.. code-block:: python

    dataname = dataset you want to use
    num_parts =  shots you what to use
    graph_list, input_dim, hid_dim = load_data4pretrain(dataname, num_parts)

- **third**: determine the pretrain method you want to use and build the task of pretrain:
.. code-block:: python

    pt = PreTrain(pre_train_method, gnn_type, input_dim, hid_dim, gln)

- **last**: run the task, get the trained model and save it
.. code-block:: python

    pt.train(graph_list, batch_size=batch_size, lr=0.01, decay=0.0001, epochs=100)

- The following codes present a simple example on how to pre-train a GNN model via GraphCL:

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

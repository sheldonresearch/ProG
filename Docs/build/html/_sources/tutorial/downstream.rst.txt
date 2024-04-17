downstream
============
we summarized all the tasks into 3 kind
::
    - Node classification
    - edge prediction
    - graph classification

To build a task, you just need to

#. ``pretrain a model and save it in approperate path``

#. ``use the same dataset and parameters of your model``

#. ``determine the prompt type you want``

#. ``build the tasker``

Here are some examples

.. code-block:: python

    from ProG.tasker import NodeTask, LinkTask, GraphTask
    from ProG.prompt import GPF, GPF_plus, GPPTPrompt, GPrompt, LightPrompt

    tasker = NodeTask(pre_train_model_path = 'None',
                  dataset_name = 'Cora', num_layer = 3, gnn_type = 'GCN', prompt_type = 'gpf', shot_num = 5)

    # tasker = LinkTask(pre_train_model_path = './pre_trained_gnn/Cora.Edgepred_Gprompt.GCN.pth',
    #                      dataset_name = 'Cora', gnn_type = 'GAT', prompt_type = 'None')

    # tasker = GraphTask(pre_train_model_path = './pre_trained_gnn/MUTAG.SimGRACE.GCN.128hidden_dim.pth',
    #                      dataset_name = 'MUTAG', gnn_type = 'GCN', prompt_type = 'gpf', shot_num = 50)

    # tasker = GraphTask(pre_train_model_path = 'None',
    #                      dataset_name = 'MUTAG', gnn_type = 'GCN', prompt_type = 'ProG', shot_num = 20)

    # tasker = GraphTask(pre_train_model_path = 'None',
    #                      dataset_name = 'ENZYMES', gnn_type = 'GCN', prompt_type = 'None', shot_num = 50)
    tasker.run()

.. note::
    - Kindly note that the comparison takes the same pre-trained pth.
    - The absolute value of performance won't mean much because the final results may vary depending on different pre-training states.
    - It would be more interesting to see the relative performance with other training paradigms.
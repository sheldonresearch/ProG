compared with the raw code, this 0.1 version has EXTREAMLY HUGE CHANGES, including but not limited to:

- totally rewrite the whole project, the changed code takes up >80% of the original version.
- simplify the code
- rename some functions and files for more readable experience.
- adopt torchmetrics for automatic accumulation over batches in the evaluation stage (e.g. Acc, F1 etc), gradually
  remove sklearn.metrics in the original version
- more clear prompt module. in the previous version, there are at least two different implementations for prompt,which
  is very messy. Here we remove all these trash and unify them with a LightPrompt and a HeavyPrompt.
- support batch training and testing in function: meta_test_adam


An example to see how much degree we have reduced for the code simplification:

Pipeline before, after





evaluated results from this version of code

```
Multi-class node classification (100-shots)

                      |      CiteSeer     |
                      |  ACC  | Macro-F1  |
==========================================|
reported in the paper | 80.50 |   80.05   |
(Prompt)              |                   |
------------------------------------------|
this version code     | 81.00 |   --      |
(Prompt)              |   (run one time)  | 
==========================================|
reported in the paper | 80.00 ｜  80.05   ｜
(Prompt w/o h)        |                   ｜
------------------------------------------|
this version code     | 79.78 ｜  80.01   ｜
(Prompt w/o h)        |   (run one time)  ｜
==========================================|
--: hasn't implemented batch F1 in this version
```









Future TODO list (Call for partner)
- remove our self-implemented MAML module, replace it with a third-party meta library such as learn2learn or Torchmeta
- support sparse training
- support GPU
- support True Batch computing
- support GIN and more GNNs
- support more Pre-train methods such as GraphGPT
- test on large-scale
- support distributed computing
- support more tasks and data sets





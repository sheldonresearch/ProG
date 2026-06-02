PRETRAIN_TYPES = ["DGI", "GraphMAE", "Edgepred_GPPT", "Edgepred_Gprompt", "GraphCL", "SimGRACE"]


NODE_TASKS = [
    "PubMed",
    "CiteSeer",
    "Cora",
    "Computers",
    "Photo",
    "Reddit",
    "WikiCS",
    "Flickr",
    "ogbn-arxiv",
    "Actor",
    "Texas",
    "Wisconsin",
]

GRAPH_TASKS = [
    "MUTAG",
    "ENZYMES",
    "COLLAB",
    "PROTEINS",
    "IMDB-BINARY",
    "REDDIT-BINARY",
    "COX2",
    "BZR",
    "PTC_MR",
    "ogbg-ppa",
    "DD",
]

PROMPT_TYPES = ["None", "GPPT", "All-in-one", "Gprompt", "GPF", "GPF-plus"]


# ---- LinkTask dataset catalog ------------------------------------------------
#
# LinkTask single-graph LP datasets — every NODE_TASKS dataset is structurally
# usable (any graph has edges), but the largest ones blow up dense negative
# sampling and the per-epoch encoder pass. They're excluded from the default
# sweep list and require an explicit opt-in. Same gating rationale as the
# `num_iter = 1` / `batch_size = 512` special cases in bench.do_config_bench
# for ogbn-arxiv / Flickr.
LINK_TASK_LARGE_NODE_DATASETS = ["Reddit", "Flickr", "ogbn-arxiv"]
LINK_TASK_SINGLE_GRAPH_DATASETS = [d for d in NODE_TASKS if d not in LINK_TASK_LARGE_NODE_DATASETS]

# LinkTask multi-graph LP datasets — TUDatasets / OGB-graph batched into a
# single mega-graph for link prediction. The large ones (COLLAB, DD,
# REDDIT-BINARY, ogbg-ppa) currently fail single-Batch loading even in the
# Edgepred_* pretrains (see prompt_graph/pretrain/Edgepred_GPPT.py), so they
# need the per-graph mini-batch loader before they're usable here.
LINK_TASK_LARGE_GRAPH_DATASETS = ["COLLAB", "DD", "REDDIT-BINARY", "ogbg-ppa"]
LINK_TASK_MULTI_GRAPH_DATASETS = [d for d in GRAPH_TASKS if d not in LINK_TASK_LARGE_GRAPH_DATASETS]

LINK_TASKS = LINK_TASK_SINGLE_GRAPH_DATASETS + LINK_TASK_MULTI_GRAPH_DATASETS

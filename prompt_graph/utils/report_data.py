from dataclasses import dataclass, field
from typing import Optional




@dataclass
class ConfigBenchResult:
    pretrain_task_type:str
    dataset_name:str
    prompt_type:str

    best_params: dict[str, float] = field(
        default_factory=lambda: {}
    )

    best_loss:float = float('inf')
    final_acc_mean:float = float('inf')
    final_acc_std:float = float('inf')
    final_f1_mean:float = float('inf')
    final_f1_std:float = float('inf')
    final_roc_mean:float = float('inf')
    final_roc_std:float = float('inf')
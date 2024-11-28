
import os
from Utility import config_resoler

import copy

from prompt_graph.utils.get_args import get_args_by_call

from pre_train import get_pretrain_task_delegate

import json

import traceback



class BatchRunPretrain:
    def __init__(self,predefined_pretrains) -> None:
        

        self._predefined_pretrains:list[dict] = sorted(predefined_pretrains,key=lambda x: json.dumps(x,ensure_ascii=False))


    
    def start(self,device = 1, skip_error:bool = False):
        
        BASIC_ARG_DICT = {
            "epochs":1000,
            "num_layer":2,
            "lr":0.02,
            "decay":2e-6,
            "seed":42,
            "device":device,
        }
        

        for pt_config_dict in self._predefined_pretrains:

            certain_config = copy.deepcopy(BASIC_ARG_DICT)

            certain_config.update(pt_config_dict)

            args = get_args_by_call(**certain_config)

            saved_path = config_resoler.GetSavedPretrainModelPath(args)
            if os.path.isfile(saved_path):
                print(f"Ignore existed pretrain file {saved_path}")
                continue

            
            try:
                print(f"Starting {args}.")
                tasker = get_pretrain_task_delegate(args=args)
                
                print(tasker.pretrain())
            except KeyboardInterrupt:
                print("\nProcess interrupted by user.")
                if input("Do you wish to stop? y/n [y]:").lower().startswith("y"):
                    break
            except Exception as e:
                traceback.print_exc()
                print(f"Error config: {certain_config}")
                if not skip_error and input("Do you wish to stop? y/n [n]:").lower().startswith("y"):
                    break


if __name__ == "__main__":
    _pd = config_resoler.ConfigResolver.resolve_config_from_dir("./Experiment/provided_pt_models")

    _pd = [item for item in _pd if item.get("pretrain_task") not  in ("Edgepred_Gprompt",) and item.get("dataset_name") not in ("COLLAB",)]
    _runner = BatchRunPretrain(_pd)

    _runner.start(device=2,skip_error=True)
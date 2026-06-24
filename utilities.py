# +
import yaml
from types import SimpleNamespace

class Mapping():
    
    def __init__(self, items):        
        self.item2idx={}
        self.idx2item=[]
        
        for idx, item in enumerate(items):
            self.item2idx[item]=idx
            self.idx2item.append(item)
            
    def add(self,item):
        if item not in self.idx2item:
            self.idx2item.append(item)
            self.item2idx[item]=len(self.idx2item)-1


def flatten_dict(d):
    """
    Flatten the nested YAML configuration into the format of args.xxx
    """
    result = {}

    for key, value in d.items():
        if isinstance(value, dict):
            result.update(flatten_dict(value))
        else:
            result[key] = value

    return result

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    flat_cfg = flatten_dict(cfg)
    return SimpleNamespace(**flat_cfg), cfg


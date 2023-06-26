import asyncio
import pickle

import alpa


def to_np_arr(state):
    dic = {}
    for k, v in state.items():
        if isinstance(v, dict):
            dic[k] = to_np_arr(v)
        elif isinstance(v, alpa.device_mesh.DistributedArray):
            #  print(f"{k} shape={v.shape} sharding={v.sharding_spec}")
            asyncio.run(v.to_np_async())
            dic[k] = v._npy_value
        else:
            #  try:
            #      pickle.dumps(v)
            #  except:
            #      print(f"cannot pickle {type(v)}")
            dic[k] = v
    return dic

import copy

import jax

import alpa


def print_specs(state):
    for k, v in state.items():
        if isinstance(v, dict):
            print_specs(v)
        elif isinstance(v, alpa.device_mesh.DistributedArray):
            print(f"{k} shape={v.shape} sharding={v.sharding_spec}")
        else:
            pass


def print_specs_txt(state, keys=None):
    if keys is None:
        keys = []
    txt = ""

    for k, v in state.items():
        new_keys = copy.deepcopy(keys)
        new_keys.append(k)
        if isinstance(v, dict):
            txt += print_specs_txt(v, new_keys)
        elif isinstance(v, alpa.device_mesh.DistributedArray):
            txt += f"{new_keys} shape {v.shape} sharding {v.sharding_spec} device mesh {v.device_mesh.host_ids} {v.device_mesh.devices}\n"
        elif isinstance(v, jax.ShapeDtypeStruct):
            txt += f"{new_keys} shape {v.shape}\n"
        else:
            pass

    return txt

import torch

def get_model_para_number(model):
    total_number = 0
    for para in model.parameters():
        total_number += torch.numel(para)

    return total_number

def check_params(params, **kwargs):
    for key, val in kwargs.items():
        key_defined = (key in params.keys())
        if val is None:
            assert key_defined, 'Params must include {}'.format(key)
        elif not key_defined:
            params[key] = val

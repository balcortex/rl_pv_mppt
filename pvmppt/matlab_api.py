def compose_block_name(blocks):
    if isinstance(blocks, list):
        return "/".join(blocks)
    return blocks


def set_parameters(eng, blocks, params, nargout=0):
    block_name = compose_block_name(blocks)
    for key, value in params.items():
        if isinstance(value, list):
            param_list = []
            for val in value:
                param_list.append(key)
                param_list.append(val)
            eng.set_param(block_name, *param_list, nargout=nargout)
        else:
            eng.set_param(block_name, key, value, nargout=nargout)


def get_parameter(eng, blocks, param_to_get):
    block_name = compose_block_name(blocks)
    return eng.get_param(block_name, param_to_get)

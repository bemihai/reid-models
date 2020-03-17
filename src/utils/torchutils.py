import pickle
import os
import warnings
import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict
from .utils import mkdir_if_missing


def save_checkpoint(state, save_dir, remove_module_from_keys=False):
    r"""
    Save checkpoint.

    Args:
        state (dict): dictionary containing model, epoch, state_dict, optimizer, scheduler, metric, etc.
        save_dir (str): directory to save checkpoint.
        remove_module_from_keys (bool, optional): whether to remove "module."
            from layer names. Default is False.
    """
    assert 'state_dict' in state.keys()
    mkdir_if_missing(save_dir)
    if remove_module_from_keys:
        state_dict = state['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]  # remove 'module.' in state_dict's keys
            new_state_dict[k] = v
        state['state_dict'] = new_state_dict
    # save checkpoint
    model_name = state['model'] if 'model' in state.keys() else 'model'
    epoch_idx = str(state['epoch']) if 'epoch' in state.keys() else '1'
    fpath = os.path.join(save_dir, model_name + '-epoch-' + epoch_idx + '.pth.tar')
    torch.save(state, fpath)
    print('Checkpoint saved to "{}"'.format(fpath))


def load_checkpoint(fpath):
    r"""
    Load checkpoint.

    Args:
        fpath (str): path to checkpoint.
    """
    if fpath is None:
        raise ValueError('File path is None')
    if not os.path.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(fpath, pickle_module=pickle, map_location=map_location)
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint


def open_all_layers(model):
    """ Open all layers in a model for training. """
    model.train()
    for p in model.parameters():
        p.requires_grad = True


def open_specified_layers(model, open_layers):
    r"""
    Open specified layers in a model for training while keeping other layers frozen.

    Args:
        model (nn.Module): model.
        open_layers (str or list): layers open for training.
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    if isinstance(open_layers, str):
        open_layers = [open_layers]

    for layer in open_layers:
        assert hasattr(
            model, layer
        ), '"{}" is not an attribute of the model, please provide the correct name'.format(layer)

    for name, module in model.named_children():
        if name in open_layers:
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        else:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


def load_pretrained_weights(model, fpath, remove_module_from_keys=False):
    r"""
    Load pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        fpath (str): path to pretrained weights (or checkpoint).
        remove_module_from_keys (bool, optional): whether to remove "module."
            from layer names. Default is False.
    """
    checkpoint = load_checkpoint(fpath)
    if 'state_dict' in checkpoint.keys():
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if remove_module_from_keys and k.startswith('module.'):
            k = k[7:]  # remove "module." from layers names

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn('The pretrained weights "{}" cannot be loaded, please check the key names manually '
                      '(** ignored and continue **)'.format(fpath))
    else:
        print('Successfully loaded pretrained weights from "{}"'.format(fpath))
        if len(discarded_layers) > 0:
            print('** The following layers are discarded due to unmatched keys or layer size: {}'
                  .format(discarded_layers))

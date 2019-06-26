"""variable.py: Utilities around handling variables."""
# by Hiroaki Hayashi

import torch
from torch.autograd import Variable

def cudaw(obj, use_cuda=True):
    """Call .cuda on the object if cuda.is_available.
    by @jayanthkoushik.
    """
    if torch.cuda.is_available() and use_cuda:
        obj = obj.cuda()
    return obj


def var2np(var):
    """Convert a torch Variable to a numpy array.
    by @jayanthkoushik.
    """
    return var.data.cpu().numpy()


def update_var(var, data):
    """Update a torch variable with the given data, and return the updated
    version. Create if var is None.
    by @jayanthkoushik.
    """
    if var is None:
        var = cudaw(Variable(data))
    else:
        var.data[:] = data
    return var


def total_parameter_size(model):
    """look up state dict of the model and returns the total number of
    parameters.

    Parameters:
        model (nn.Module): pytorch model.
    Returns:
        the number of parameters including custom weight matrices.
    """
    return sum([w.nelement() for k, w in model.state_dict().items()])
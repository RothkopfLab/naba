import dill

from naba.nn import ActionNetwork


def save(filename: str, model: ActionNetwork):
    """ Save the model to a pickle file

    Args:
        filename: filename
        model: ActionNetwork

    Returns:
        None
    """
    dill.dump(model, open(filename, "wb"))


def load(filename: str) -> ActionNetwork:
    """ Load a model from a pickle file

    Args:
        filename: filename

    Returns:
        ActionNetwork
    """
    return dill.load(open(filename, "rb"))

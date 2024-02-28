import numpy,os,pandas
import datetime,tqdm


def get_module_path(dir: list[str]) -> str:
    """
    Returns the path to a subdirectory named 'dir' relative to the currently executed script.

    Args:
        dir (str): path to the subdirectory.

    Returns:
        str: Absolute path to the specified subdirectory.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),*dir)

def now()-> int:
    """
    Returns the current timestamp as an integer.

    Returns:
        int: Current timestamp (number of seconds since the epoch).
    """
    return int(datetime.datetime.now().timestamp())


def unfold(d: dict) -> dict:
    """
    Unfolds a nested dictionary by appending the values of inner dictionaries to the outer dictionary.

    Args:
        d (dict): Input dictionary with nested dictionaries.

    Returns:
        dict: Unfolded dictionary with concatenated keys.
    
    Example:
        >>> d = {'a':1,'b':{'c':2,'d':4}}
        >>> unfold(d)
        {'a': 1, 'b_c': 2, 'b_d': 4}
    """
    new_d = {}
    for k in d:
        if hasattr(d[k],'keys'):
            for j in d[k]:
                new_d[f'{k}_{j}'] = d[k][j]
        else:
            new_d[k] = d[k]
    return new_d

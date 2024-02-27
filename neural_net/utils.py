import numpy,os,pandas
import datetime,tqdm


get_module_path = lambda dir : os.path.join(os.path.dirname(os.path.abspath(__file__)),*dir)

now = lambda : int(datetime.datetime.now().timestamp())


def unfold(d):
    new_d = {}
    for k in d:
        if hasattr(d[k],'keys'):
            for j in d[k]:
                new_d[f'{k}_{j}'] = d[k][j]
        else:
            new_d[k] = d[k]
    return new_d

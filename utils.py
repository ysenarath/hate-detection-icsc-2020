from os import environ
from os.path import join

__all__ = [
    'scratch_path'
]


def scratch_path(path):
    if environ.get('SCRATCH') is not None:
        return join(environ.get('SCRATCH'), path)
    else:
        return path

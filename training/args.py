_GLOBAL_ARGS = None

def get_args():
    return _GLOBAL_ARGS

def set_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args
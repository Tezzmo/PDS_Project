import os


def get_ml_path():
    if os.path.isdir(os.path.join(os.getcwd(), 'ml-models')):
        return os.path.join(os.getcwd(), 'ml-models')
    elif os.path.isdir(os.path.join(os.getcwd(), "../ml-models")):
        return os.path.join(os.getcwd(), "../ml-models")
    else:
        raise FileNotFoundError


def get_gejson_path():
    if os.path.isdir(os.path.join(os.getcwd(), 'data')):
        return os.path.join(os.getcwd(), 'data')
    elif os.path.isdir(os.path.join(os.getcwd(), "../data")):
        return os.path.join(os.getcwd(), "../data")
    else:
        raise FileNotFoundError

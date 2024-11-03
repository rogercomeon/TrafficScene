import json


from loader.LF_dataset_loader import LFdatasetLoader
def get_loader(name):
    """get_loader

    :param name:
    """
    return {"LF_dataset":LFdatasetLoader}[name]
  

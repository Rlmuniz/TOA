import os
import yaml

DATA_PATH = os.path.join(os.path.dirname(__file__), 'airplanes')


class Airplane:
    pass


def dict2obj(item, klss=Airplane):
    # checking whether object d is a
    # instance of class list
    if isinstance(item, list):
        item = [dict2obj(x) for x in item]

    # if d is not a instance of dict then
    # directly object is returned
    if not isinstance(item, dict):
        return item

    obj = klss()

    for k in item:
        obj.__dict__[k] = dict2obj(item[k])

    return obj


def load_airplane_data(id, datapath):
    """Load airplane's data"""
    filepath = os.path.join(datapath, f"{id.lower()}.yaml")
    try:
        with open(filepath) as file:
            data = yaml.full_load(file)
    except FileNotFoundError:
        print(f"There is no airplane with the following id: {id}")
    else:
        return data


def get_airplane_data(id, datapath=DATA_PATH):
    data = load_airplane_data(id, datapath)
    return dict2obj(data)

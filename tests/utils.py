import numpy
import json

def get_case_data_dict(case_path):
    with open(case_path, 'r') as fp:
        case_data = json.load(fp)

    for key, value in case_data.items():
        if isinstance(value, list):
            value = numpy.array(value)
            case_data[key] = value

    return case_data

def drop_rep_zeros(old_arr: numpy.ndarray) -> numpy.ndarray:
    nozero_yet = True
    new_arr = []
    for el in old_arr:
        if nozero_yet and el == 0:
            nozero_yet = False
            new_arr.append(el)
        elif el != 0:
            new_arr.append(el)
    new_arr = numpy.array(new_arr)
    return new_arr
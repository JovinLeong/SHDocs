import json
from datetime import datetime

def dump_dict_to_json(my_dict, filename):
    with open(filename, 'w') as f:
        json.dump(my_dict, f, indent=4)
        
def get_timenow():
    return str(datetime.now()).split('.')[0].replace('-','').replace(' ', '_').replace(':', '')

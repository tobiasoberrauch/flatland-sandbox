import json

from lib import find_index_in_list

with open('tmp/agent_targets/data-1635000858.52632.json') as fp:
    data = json.load(fp)
    
    print(find_index_in_list(data, 1.0))
    
    fp.close()
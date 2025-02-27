from pathlib import Path 
import json


def write_json(data, path, file_name, indent, overwrite = False):
    
    if '.json' not in file_name:
        file_name = file_name + '.json '
    
    if not overwrite:
        if Path(path, file_name).exists():
            with open(Path(path, file_name), 'r') as f:
                json_content = json.load(f)
                json_content.append(data)
    else:
        if not isinstance(data, list):
         json_content = [data]
        

    with open(Path(path, file_name), 'w') as f: 
        json.dump(json_content, f, indent=indent) #separators=(',', ',\n'))
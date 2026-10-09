from pathlib import Path
import numpy as np  






def parse_line (line ) :
    parts = line.split()
     
    label = int(parts[0])

    features =[]  

    for item in parts[1:] : 
        index , value = item.split(":")
        features.append(float(value))
    return label , features 




def load_batches(data_path):
        
    files = [ f for f in data_path.iterdir() if f.is_file()] 
    batches = {}
    for file in files : 
        data = file 
        with open ( data  , 'r') as dat  : 
            lines  = dat.readlines()
            features = []
            labels = []
            for line in lines : 
                lab, fea = parse_line(line)
                features.append(fea )
                labels.append(lab)
            features = np.array(features)
            labels = np.array(labels )
            batches[file.stem] =  {"features" :  features , "labels" : labels}


    
    sorted_batches = sorted(
        batches.items(),
        key=lambda item: int(item[0].replace("batch", ""))
    )
    return dict(sorted_batches )


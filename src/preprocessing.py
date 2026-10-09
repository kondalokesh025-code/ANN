import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


def prepare_data(batches ) : 



    sorted_batches = sorted(
        batches.items(),
        key=lambda item: int(item[0].replace("batch", ""))
    )

        
    X_train = []
    y_train = []
    i = 0 
    for   i , (batch_name , batch_data) in enumerate(sorted_batches) :
        if i == 6 : 
            break 
        
        X_train.append(np.array(batch_data["features"]))
        y_train.append(np.array(batch_data["labels"]))

    X_train = np.concatenate(X_train )
    y_train = np.concatenate(y_train)





    batch_name, batch_data = sorted_batches[6]


    X_val = np.array(batch_data["features"])
    y_val = np.array(batch_data["labels"])


    X_test_8 = np.array(sorted_batches[7][1]["features"])
    y_test_8 = np.array(sorted_batches[7][1]["labels"])

    X_test_9 = np.array(sorted_batches[8][1]["features"])
    y_test_9 = np.array(sorted_batches[8][1]["labels"])

    X_test_10 = np.array(sorted_batches[9][1]["features"])
    y_test_10 = np.array(sorted_batches[9][1]["labels"])



    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val )
    X_test_8 = scaler.transform(X_test_8)
    X_test_9 = scaler.transform(X_test_9)
    X_test_10 = scaler.transform(X_test_10)




    y_train = y_train - 1
    y_val = y_val - 1
    y_test_8 = y_test_8 - 1
    y_test_9 = y_test_9 - 1
    y_test_10 = y_test_10 - 1


    X_train = torch.tensor(X_train , dtype = torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.int64)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.int64)

    X_test_8 = torch.tensor(X_test_8, dtype=torch.float32)
    y_test_8 = torch.tensor(y_test_8, dtype=torch.int64)

    X_test_9 = torch.tensor(X_test_9, dtype=torch.float32)
    y_test_9 = torch.tensor(y_test_9, dtype=torch.int64)

    X_test_10 = torch.tensor(X_test_10, dtype=torch.float32)
    y_test_10 = torch.tensor(y_test_10, dtype=torch.int64)


    train_dataset = TensorDataset(X_train , y_train)
    val_dataset = TensorDataset(X_val , y_val)

    train_loader = DataLoader(
        train_dataset ,
        batch_size = 64 , 
        shuffle = True 
    )
    val_loader = DataLoader(
        val_dataset , 
        batch_size = 64 , 
        shuffle = False 
    )
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test_8": X_test_8,
        "y_test_8": y_test_8,
        "X_test_9": X_test_9,
        "y_test_9": y_test_9,
        "X_test_10": X_test_10,
        "y_test_10": y_test_10,
        "scaler": scaler,
        "train_loader": train_loader,
        "val_loader": val_loader
    }

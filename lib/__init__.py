import torch
import torch.nn as nn

from datetime import datetime
from typing import List, Union, Type
import os

from lib.module import get_dataset, model_test


def training(input_dataset:List[List[str]], net_type:Type[nn.Module], epoch:int) -> str:
    ## Data setting
    x_train, x_valid, y_train, y_valid = get_dataset(input_dataset)

    ## Training
    model = net_type()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for ep in range(1, epoch + 1):
        ## Forward pass
        y_pred_train = model(x_train)
        y_pred_val = model(x_valid)

        ## Calculate loss
        loss_train = criterion(y_pred_train, y_train)
        loss_val = criterion(y_pred_val, y_valid)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        if ep % 100 == 0:
            print(f"Epoch {ep}, Training loss : {loss_train.item()}, Validation Loss: {loss_val.item()}")

    ## Model save
    _d = datetime.now().strftime("%Y%m%d%H%M%S")
    save_file_name = f"{net_type.__name__}_{_d}_train{loss_train.item():.2f}_val{loss_val.item():.2f}.pth"
    torch.save(model.state_dict(), save_file_name)
    print(f"train DONE. save {save_file_name}.")

    return save_file_name


def running(net_type:Type[nn.Module], pth_file: str, data_list: List[str], y_data_list: Union[List[str], None]=None):
    ## Parameter setting
    if not os.path.exists(pth_file):
        print(f"{pth_file} is not exist.")
        return False
    if y_data_list is None:
        y_data_list = [None for x in range(len(data_list))]

    ## Model run
    model = net_type()
    model.load_state_dict(torch.load(pth_file))
    for data, y_data in zip(data_list, y_data_list):
        model_test(model, data, y_data)
        
    print("run model DONE")
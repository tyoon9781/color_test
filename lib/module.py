import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from datetime import datetime
from typing import List, Tuple, Union
import os


def _hex_to_rgb(hex_color:str) -> np.ndarray:
    """
    "#FFFFFF" -> np.array([255, 255, 255])
    """
    if hex_color[0] == "#":
        hex_color = hex_color[1:]
    if len(hex_color) != 6:
        print(f"hex color format error ({hex_color})")
        raise ValueError
    return np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)], dtype=np.uint8)


def _rgb_to_hex(rgb_color: np.ndarray) -> str:
    """
    np.array([255, 255, 255]) -> "#FFFFFF"
    """
    return "#" + "".join(["{:02x}".format(int(c)) for c in rgb_color])


def _get_dataset(dataset: List) -> Tuple[np.ndarray, np.ndarray]:
    dataset = np.array(dataset)
    X = np.array([_hex_to_rgb(x) for x in dataset[:, 0]])
    Y = np.array([_hex_to_rgb(y) for y in dataset[:, 1]])
    return (X, Y)


def _convert_tensor(data: np.ndarray) -> torch.Tensor:
    return torch.tensor(data, dtype=torch.float)


def _convert_tensor_list(data_list: List[np.ndarray]) -> List[torch.Tensor]:
    result_list = []
    for data in data_list:
        result_list.append(_convert_tensor(data))
    return result_list


def _model_test(model: nn.Module, test_data: str, y_data:str="no data"):
    convert_test_data = _convert_tensor(_hex_to_rgb(test_data))
    predict = model(convert_test_data)
    predict_hex = _rgb_to_hex(predict.data.numpy())
    print(f"input data : {test_data}, predict data : {predict_hex}, answer data : {y_data}")


def training(input_dataset:List[List[str]], model:nn.Module, epoch:int) -> str:
    ## data setting
    X, Y = _get_dataset(input_dataset)
    result = train_test_split(X, Y, test_size=0.25, random_state=42)
    (x_train, x_val, y_train, y_val) = _convert_tensor_list(result)

    ## train
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for ep in range(1, epoch + 1):
        ## forward pass
        y_pred_train = model(x_train)
        y_pred_val = model(x_val)

        ## calculate loss
        loss_train = criterion(y_pred_train, y_train)
        loss_val = criterion(y_pred_val, y_val)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        if ep % 100 == 0:
            print(f"Epoch {ep}, Training loss : {loss_train.item()}, Validation Loss: {loss_val.item()}")

    ## Model save
    _d = datetime.now().strftime("%Y%m%d%H%M%S")
    save_file_name = f"ColorNetV1_{_d}_val{loss_val.item():.2f}.pth"
    torch.save(model.state_dict(), save_file_name)
    print("train DONE")
    return save_file_name


def running(model:nn.Module, pth_file: str, data_list: List[str], y_data_list: Union[List[str], None]=None):
    if not os.path.exists(pth_file):
        print(f"{pth_file} is not exist.")
        return False
    if y_data_list is None:
        y_data_list = [None for x in range(len(data_list))]

    model.load_state_dict(torch.load(pth_file))
    for data, y_data in zip(data_list, y_data_list):
        _model_test(model, data, y_data)
    print("run model DONE")
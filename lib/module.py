import numpy as np
import random
import torch
import torch.nn as nn
from typing import List, Tuple


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


def _split_x_y(dataset: List) -> Tuple[np.ndarray, np.ndarray]:
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


def model_test(model: nn.Module, test_data: str, y_data:str="no data"):
    convert_test_data = _convert_tensor(_hex_to_rgb(test_data))
    predict = model(convert_test_data)
    predict_hex = _rgb_to_hex(predict.data.numpy())
    print(f"input data : {test_data}, predict data : {predict_hex}, answer data : {y_data}")


def get_dataset(input_dataset, train_ratio=0.8, seed=None):
    if seed is not None:
        random.seed(seed)

    X, Y = _split_x_y(input_dataset)
    random_list = [x for x in range(len(X))]
    random.shuffle(random_list)
    cut_index = int(len(X)*train_ratio)
    x_shape, y_shape = X.shape[1], Y.shape[1]

    train_index_list, valid_index_list = random_list[:cut_index], random_list[cut_index:]
    x_train = np.zeros((len(train_index_list), x_shape), dtype=np.uint8)
    x_valid = np.zeros((len(valid_index_list), x_shape), dtype=np.uint8)
    y_train = np.zeros((len(train_index_list), y_shape), dtype=np.uint8)
    y_valid = np.zeros((len(valid_index_list), y_shape), dtype=np.uint8)

    for i, train_index in enumerate(train_index_list):
        x_train[i], y_train[i] = X[train_index], Y[train_index]
    for i, valid_index in enumerate(valid_index_list):
        x_valid[i], y_valid[i] = X[valid_index], Y[valid_index]

    return _convert_tensor_list([x_train, x_valid, y_train, y_valid])


# def training(input_dataset:List[List[str]], net, epoch:int) -> str:
#     ## data setting
#     x_train, x_valid, y_train, y_valid = _get_dataset(input_dataset)

#     ## model set
#     model:nn.Module = net()

#     ## train
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     for ep in range(1, epoch + 1):
#         ## forward pass
#         y_pred_train = model(x_train)
#         y_pred_val = model(x_valid)

#         ## calculate loss
#         loss_train = criterion(y_pred_train, y_train)
#         loss_val = criterion(y_pred_val, y_valid)
#         optimizer.zero_grad()
#         loss_train.backward()
#         optimizer.step()

#         if ep % 100 == 0:
#             print(f"Epoch {ep}, Training loss : {loss_train.item()}, Validation Loss: {loss_val.item()}")

#     ## Model save
#     _d = datetime.now().strftime("%Y%m%d%H%M%S")
#     save_file_name = f"{net.__name__}_{_d}_train{loss_train.item():.2f}_val{loss_val.item():.2f}.pth"
#     torch.save(model.state_dict(), save_file_name)
#     print("train DONE")
#     return save_file_name


# def running(net, pth_file: str, data_list: List[str], y_data_list: Union[List[str], None]=None):
#     ## model set
#     model:nn.Module = net()

#     if not os.path.exists(pth_file):
#         print(f"{pth_file} is not exist.")
#         return False
#     if y_data_list is None:
#         y_data_list = [None for x in range(len(data_list))]

#     model.load_state_dict(torch.load(pth_file))
#     for data, y_data in zip(data_list, y_data_list):
#         _model_test(model, data, y_data)
#     print("run model DONE")
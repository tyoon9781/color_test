
from lib import module as md
from lib import dataset
from lib import models

def main():
    ## training
    model = models.ColorNetV1()
    pth_file = md.training(dataset.red_green_color_list, model, 10000)

    ## sample data test
    test_data_list = ["#BDA74B", "#CC28A9", "#1DEB6B"]
    md.running(model, pth_file, test_data_list)
    print("DONE")


if __name__ == "__main__":
    main()
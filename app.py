from lib import models, dataset, training, running

def main():
    ## training
    net = models.ColorNetV2
    pth_file = training(dataset.red_green_color_list, net, 10000)

    ## sample data test
    running(net, pth_file, dataset.test_data_list)
    print("DONE")


if __name__ == "__main__":
    main()
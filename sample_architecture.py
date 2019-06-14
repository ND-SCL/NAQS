CIFAR10_6 = {}
CIFAR10_6[0] = [
    {'num_filters': 32, 'filter_height': 3, 'filter_width': 3, 'stride_height': 1, 'stride_width': 1, 'pool_size': 1},
    {'num_filters': 32, 'filter_height': 3, 'filter_width': 3, 'stride_height': 1, 'stride_width': 1, 'pool_size': 2},
    {'num_filters': 64, 'filter_height': 3, 'filter_width': 3, 'stride_height': 1, 'stride_width': 1, 'pool_size': 1},
    {'num_filters': 64, 'filter_height': 3, 'filter_width': 3, 'stride_height': 1, 'stride_width': 1, 'pool_size': 2},
    {'num_filters': 128, 'filter_height': 3, 'filter_width': 3, 'stride_height': 1, 'stride_width': 1, 'pool_size': 1},
    {'num_filters': 128, 'filter_height': 3, 'filter_width': 3, 'stride_height': 1, 'stride_width': 1, 'pool_size': 2}]

CIFAR10_6[1] = [
    {'num_filters': 32, 'filter_height': 3, 'filter_width': 3, 'stride_height': 1, 'stride_width': 1, 'pool_size': 1},
    {'num_filters': 128, 'filter_height': 3, 'filter_width': 5, 'stride_height': 1, 'stride_width': 1, 'pool_size': 1},
    {'num_filters': 64, 'filter_height': 3, 'filter_width': 3, 'stride_height': 1, 'stride_width': 1, 'pool_size': 2},
    {'num_filters': 96, 'filter_height': 5, 'filter_width': 5, 'stride_height': 1, 'stride_width': 1, 'pool_size': 2},
    {'num_filters': 128, 'filter_height': 3, 'filter_width': 7, 'stride_height': 1, 'stride_width': 1, 'pool_size': 1},
    {'num_filters': 64, 'filter_height': 3, 'filter_width': 7, 'stride_height': 1, 'stride_width': 2, 'pool_size': 1}]

CIFAR10_6[2] = [
    {'num_filters': 48, 'filter_height': 3, 'filter_width': 3, 'stride_height': 1, 'stride_width': 1, 'pool_size': 1},
    {'num_filters': 64, 'filter_height': 1, 'filter_width': 5, 'stride_height': 1, 'stride_width': 1, 'pool_size': 1},
    {'num_filters': 64, 'filter_height': 5, 'filter_width': 3, 'stride_height': 1, 'stride_width': 1, 'pool_size': 2},
    {'num_filters': 96, 'filter_height': 5, 'filter_width': 5, 'stride_height': 1, 'stride_width': 1, 'pool_size': 2},
    {'num_filters': 96, 'filter_height': 3, 'filter_width': 3, 'stride_height': 1, 'stride_width': 1, 'pool_size': 1},
    {'num_filters': 96, 'filter_height': 3, 'filter_width': 7, 'stride_height': 1, 'stride_width': 2, 'pool_size': 1}]

CIFAR10_6[3] = [
    {'num_filters': 48, 'filter_height': 3, 'filter_width': 3, 'stride_height': 1, 'stride_width': 1, 'pool_size': 1},
    {'num_filters': 64, 'filter_height': 3, 'filter_width': 5, 'stride_height': 1, 'stride_width': 1, 'pool_size': 1},
    {'num_filters': 64, 'filter_height': 5, 'filter_width': 3, 'stride_height': 1, 'stride_width': 1, 'pool_size': 2},
    {'num_filters': 96, 'filter_height': 5, 'filter_width': 5, 'stride_height': 1, 'stride_width': 1, 'pool_size': 2},
    {'num_filters': 128, 'filter_height': 3, 'filter_width': 3, 'stride_height': 1, 'stride_width': 1, 'pool_size': 1},
    {'num_filters': 96, 'filter_height': 3, 'filter_width': 7, 'stride_height': 1, 'stride_width': 2, 'pool_size': 1}]

NAS15 = [
    {'filter_height': 3, 'filter_width': 3, 'num_filters': 36,  # 0
     'anchor_point': []},
    {'filter_height': 3, 'filter_width': 3, 'num_filters': 48,  # 1
     'anchor_point': [1]},
    {'filter_height': 3, 'filter_width': 3, 'num_filters': 36,  # 2
     'anchor_point': [1, 1]},
    {'filter_height': 5, 'filter_width': 5, 'num_filters': 36,  # 3
     'anchor_point': [1, 1, 1]},
    {'filter_height': 3, 'filter_width': 7, 'num_filters': 48,  # 4
     'anchor_point': [0, 0, 1, 1]},
    {'filter_height': 7, 'filter_width': 7, 'num_filters': 48,  # 5
     'anchor_point': [0, 1, 1, 1, 1]},
    {'filter_height': 7, 'filter_width': 7, 'num_filters': 48,  # 6
     'anchor_point': [0, 1, 1, 1, 1, 1]},
    {'filter_height': 7, 'filter_width': 3, 'num_filters': 36,  # 7
     'anchor_point': [1, 0, 0, 0, 0, 1, 1]},
    {'filter_height': 7, 'filter_width': 1, 'num_filters': 36,  # 8
     'anchor_point': [1, 0, 0, 0, 1, 1, 0, 1]},
    {'filter_height': 7, 'filter_width': 7, 'num_filters': 36,  # 9
     'anchor_point': [1, 0, 1, 1, 1, 1, 1, 1, 1]},
    {'filter_height': 5, 'filter_width': 7, 'num_filters': 36,  # 10
     'anchor_point': [1, 1, 0, 0, 1, 1, 1, 1, 1, 1]},
    {'filter_height': 7, 'filter_width': 7, 'num_filters': 48,  # 11
     'anchor_point': [1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1]},
    {'filter_height': 7, 'filter_width': 5, 'num_filters': 48,  # 12
     'anchor_point': [1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]},
    {'filter_height': 7, 'filter_width': 5, 'num_filters': 48,  # 13
     'anchor_point': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1]},
    {'filter_height': 7, 'filter_width': 5, 'num_filters': 48,  # 14
     'anchor_point': [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1]}]

SIMPLE = [
    {'num_filters': 32, 'filter_height': 3, 'filter_width': 3,
     'pool_size': 1},
    {'num_filters': 32, 'filter_height': 3, 'filter_width': 3,
     'pool_size': 2},
    {'num_filters': 64, 'filter_height': 3, 'filter_width': 3,
     'pool_size': 1},
    {'num_filters': 64, 'filter_height': 3, 'filter_width': 3,
     'pool_size': 2},
    {'num_filters': 128, 'filter_height': 3, 'filter_width': 3,
     'pool_size': 1},
    {'num_filters': 128, 'filter_height': 3, 'filter_width': 3,
     'pool_size': 2}]


if __name__ == '__main__':
    import torch
    import data
    import child
    import backend
    import time

    dataset = 'CIFAR10'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data = data.get_data(
        dataset, device, shuffle=True, batch_size=128)
    input_shape, num_classes = data.get_info(dataset)
    model, optimizer = child.get_model(
            input_shape, NAS15, num_classes, device
            )
    print(model.graph)
    start = time.time()
    backend.fit(
        model, optimizer,
        train_data, val_data,
        epochs=10,
        verbose=True,
        early_stop=False
        )
    end = time.time()
    print(end-start)

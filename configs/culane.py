# DATA
dataset = 'CULane'
data_root = "/media/ros/A666B94D66B91F4D/ros/test_port/Labelme2Culane/gen5"

# TRAIN
epoch = 200
batch_size = 16
optimizer = 'Adam'  # ['SGD','Adam']
learning_rate = 4e-4
weight_decay = 1e-4
momentum = 0.9

scheduler = 'cos'  # ['multi', 'cos']
# steps = [100,150]
gamma = 0.1
warmup = 'linear'
warmup_iters = 100  # 看一下干啥的 695   100   预热学习，一开始用一个较小的学习率防止一开始的学习率过大导致的不稳定

# NETWORK
use_aux = True
griding_num = 200
backbone = '18'

# LOSS
sim_loss_w = 0.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = "/media/ros/A666B94D66B91F4D/ros/test_port/Labelme2Culane/gen5/logs"


# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = "/media/ros/A666B94D66B91F4D/ros/test_port/Labelme2Culane/gen5/logs/20220824_193213_lr_4e-04_b_16/ep182.pth"
test_work_dir = None

num_lanes = 4

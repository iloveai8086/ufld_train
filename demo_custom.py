"""
2022.04.20
author:alian
车道线检测
测试自定义的数据集，并保存成检测结果图
H,W：原图尺寸；h:行锚框数，w:单元格数，C：车道线数
"""
# 导入项目源码中的文件
from model.model import parsingNet
from utils.dist_utils import dist_print
from data.constant import tusimple_row_anchor,culane_row_anchor
# 导入库
import scipy.special, tqdm
import torchvision.transforms as transforms
from PIL import Image
import os, glob, cv2, argparse
import numpy as np
import torch.utils.data


class TestDataset(torch.utils.data.Dataset):  # 加载测试数据集----------------------------------------------------------
    def __init__(self, path, img_transform=None):
        super(TestDataset, self).__init__()
        self.path = path
        self.img_transform = img_transform
        self.img_list = glob.glob('%s/*.jpg' % self.path)

    def __getitem__(self, index):
        name = glob.glob('%s/*.jpg' % self.path)[index]
        img = Image.open(name)

        if self.img_transform is not None:
            img = self.img_transform(img)
        return img, name

    def __len__(self):
        return len(self.img_list)


def parse_opt():  # 参数指定-------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='18', help='骨干网络')
    parser.add_argument('--model', type=str,
                        default="/media/ros/A666B94D66B91F4D/ros/test_port/Labelme2Culane/gen5/logs/20220824_164404_lr_4e-04_b_16/ep085.pth",
                        help='模型路径')  # 设置
    parser.add_argument('--dataset', type=str, default='my', help='数据集名称')
    parser.add_argument('--source', type=str, default='/media/ros/A666B94D66B91F4D/ros/test_port/camera/my_data_test/pic', help='测试路径')  # 设置
    parser.add_argument('--savepath', type=str, default='/media/ros/A666B94D66B91F4D/ros/test_port/camera/my_data_test/results2', help='保存路径')  # 设置
    parser.add_argument('--save_video', type=bool, default=False, help='保存为视频')
    parser.add_argument('--griding_num', type=int, default=200, help='网格数')
    parser.add_argument('--num_row_anchors', type=int, default=18, help='锚框行')
    parser.add_argument('--num_lanes', type=int, default=4, help='车道数')
    opt = parser.parse_args()
    return opt


# 执行测试---------------------------------------------------------------------------------------------------------------
def run(opt):
    dist_print('start testing...')
    backbone, model, dataset, source, savepath = opt.backbone, opt.model, opt.dataset, opt.source, opt.savepath
    save_video, griding_num, num_row_anchors, num_lanes = opt.save_video, opt.griding_num, opt.num_row_anchors, opt.num_lanes
    assert opt.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']  # 残差网络骨干
    # 网络解析（griding_num：网格数；num_row_anchors：锚框行；num_lanes：车道数）
    net = parsingNet(pretrained=False, backbone=backbone, cls_dim=(griding_num + 1, num_row_anchors, num_lanes),
                     use_aux=False).cuda()
    state_dict = torch.load(model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v
    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()
    # 图像格式统一：(288, 800)，图像张量，归一化
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # 自定义数据集
    datasets = TestDataset(source, img_transform=img_transforms)
    img_w, img_h = 1920, 1080
    row_anchor = culane_row_anchor

    for dataset in zip(datasets):  # splits：图片列表 datasets：统一格式之后的数据集
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)  # 加载数据集
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            vout = cv2.VideoWriter("my" + '.avi', fourcc, 10.0, (img_w, img_h))  # 保存结果为视频文件
        else:
            vout = None
        for i, data in enumerate(tqdm.tqdm(loader)):  # 进度条显示进度
            imgs, names = data  # imgs:图像张量，图像相对路径：
            imgs = imgs.cuda()  # 使用GPU
            with torch.no_grad():  # 测试代码不计算梯度
                pred = net(imgs)  # 模型预测 输出张量：[1,101,56,C]
            # 解析预测结果-----------------------------------------------------------------------------------------------
            out_j = pred[0].data.cpu().numpy()  # 数据类型转换成numpy [101,56,C]
            out_j = out_j[:, ::-1, :]  # 将第二维度倒着取[101,56,C]
            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)  # [100,56,C] softmax 计算（概率映射到0-1之间且沿着维度0概率总和=1）
            idx = np.arange(griding_num) + 1  # 产生 1-100
            idx = idx.reshape(-1, 1, 1)  # [100,1,1]
            loc = np.sum(prob * idx, axis=0)  # [56,C]
            out_j = np.argmax(out_j, axis=0)  # 返回最大值的索引
            loc[out_j == griding_num] = 0  # 若最大值的索引=100，则说明改行为背景，不存在车道线，归零
            out_j = loc  # [56,4]


            # 将特征图上的车道线像素坐标映射到原始图像中--------------------------------------------------------------------
            grids = np.linspace(0, 800 - 1, griding_num)  # 单元格的分布
            grid = grids[1] - grids[0]  # 单元格的间隔
            img = cv2.imdecode(np.fromfile(os.path.join(source, names[0]), dtype=np.uint8),
                               cv2.IMREAD_COLOR)  # 图像读取 （1080，1920，3）
            list_point = []  # 车道线关键像素
            for i in range(out_j.shape[1]):  # C 车道线数
                dots = []
                if np.sum(out_j[:, i] != 0) > 2:  # 车道线像素数大于2
                    for k in range(out_j.shape[0]):  # 遍历行row_anchor:56
                        if out_j[k, i] > 0:
                            point = (int(out_j[k, i] * grid * img_w / 800) - 1,
                                     int(img_h * (row_anchor[opt.num_row_anchors - 1 - k] / 288)) - 1)
                            cv2.circle(img, point, 5, (0, 0, 255), -1)  # 在原始图像描述关键点

            if save_video:
                vout.write(img)  # 保存视频结果
            else:
                # 保存检测结果图
                cv2.imwrite(os.path.join(savepath, os.path.basename(names[0])), img)
        if save_video: vout.release()


if __name__ == "__main__":
    import torch.backends.cudnn
torch.backends.cudnn.benchmark = True  # 加速
opt = parse_opt()  # 指定参数
run(opt)

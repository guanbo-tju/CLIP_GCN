import os
import scipy.io as scio
import cv2
import numpy as np

# 指定输入和输出文件夹路径
input_folder = '/home/guanbo/桌面/HeatMap/CS1/pred/mat/'
output_folder = '/home/guanbo/桌面/HeatMap/CS1/pred/mask/'

# 获取文件夹下所有的.mat文件
mat_files = [f for f in os.listdir(input_folder) if f.endswith('.mat')]

# 遍历所有.mat文件
for mat_file in mat_files:
    # 拼接.mat文件的完整路径
    mat_path = os.path.join(input_folder, mat_file)

    # 从.mat文件加载数据
    data = scio.loadmat(mat_path)

    # 获取数据中的 inst_map
    inst_map = data['inst_map']

    # 转换为 OpenCV 的 Mat 对象
    opencv_mat = np.uint8(inst_map)  # 根据实际情况可能需要调整数据类型

    # 确保 opencv_mat 是有效的图像（2D或3D数组）
    if opencv_mat.ndim == 2:
        # 对于灰度图像
        cv2.imshow('Instance Map', opencv_mat)
    elif opencv_mat.ndim == 3:
        # 对于彩色图像
        cv2.imshow('Instance Map', opencv_mat[:, :, 0])  # 显示第一个通道
    else:
        print("无效的图像格式")

    # 保存图像到指定文件夹下
    output_path = os.path.join(output_folder, f"{mat_file[:-4]}.png")  # 修改保存的文件名和格式
    cv2.imwrite(output_path, opencv_mat)

# 等待用户按键关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

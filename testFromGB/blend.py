import os
import cv2

def blend_images(heatmap_folder, original_image_folder, output_folder):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取热力图和原始图像文件列表
    heatmap_files = os.listdir(heatmap_folder)
    original_image_files = os.listdir(original_image_folder)

    # 确保文件列表排序一致
    heatmap_files.sort()
    original_image_files.sort()

    # 设置融合权重
    alpha = 0.5

    # 遍历文件列表进行融合
    for heatmap_file, original_image_file in zip(heatmap_files, original_image_files):
        # 读取热力图和原始图像
        heatmap_path = os.path.join(heatmap_folder, heatmap_file)
        original_image_path = os.path.join(original_image_folder, original_image_file)
        heatmap = cv2.imread(heatmap_path)
        original_image = cv2.imread(original_image_path)

        # 调整热力图的大小以匹配原始图像
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

        # 融合图像
        blended_image = cv2.addWeighted(original_image, alpha, heatmap, 1 - alpha, 0)

        # 保存融合后的图像
        output_path = os.path.join(output_folder, f"blend_{heatmap_file}")
        cv2.imwrite(output_path, blended_image)
        print(f"Blended Image saved to {output_path}")


# 调用函数进行融合
heatmap_folder = "/home/guanbo/ubuntu2/NatureGenetics/NC/output_images/TCGA-C4-A0F7-01Z-00-DX1.606C3AFA-F93D-4226-AA49-41B5916F9018/pred/np"
original_image_folder = "/home/guanbo/ubuntu2/NatureGenetics/NC/output_images/TCGA-C4-A0F7-01Z-00-DX1.606C3AFA-F93D-4226-AA49-41B5916F9018/Images"
output_folder = "/home/guanbo/ubuntu2/NatureGenetics/NC/output_images/TCGA-C4-A0F7-01Z-00-DX1.606C3AFA-F93D-4226-AA49-41B5916F9018/pred/blend"

blend_images(heatmap_folder, original_image_folder, output_folder)

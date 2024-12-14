from utils.datasets import LoadImagesAndLabels



pat = '../pklot_dataset/test/'
dataset = LoadImagesAndLabels(path=pat, img_size=640)
img, labels, paths, shapes = dataset[0]
print(labels)  # 检查是否包含归一化的 angle 列

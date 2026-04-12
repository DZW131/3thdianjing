import numpy as np
import torch
import cv2
from PIL import Image
from .custom_transforms import FixedResize, Normalize, ToTensor
from torchvision import transforms

@torch.no_grad()
def cal_region_deeplab(image, R_net, device, patch_size=512)->(list, list):
    h, w = image.shape[0], image.shape[1]
    image = image[:, :, ::-1]
    this_batch = Image.fromarray(image)

    transform = transforms.Compose([
        FixedResize(size=patch_size),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()])

    this_batch = transform(this_batch)

    data_variable = this_batch.unsqueeze(0)
    data_variable = data_variable.cuda(device)
    result = R_net(data_variable)
    result = transforms.Resize((h, w))(result)
    result = torch.softmax(result, dim=1).squeeze()
    result = result.permute(1, 2, 0)
    result = torch.argmax(result, -1)
    result = result.detach().cpu().numpy()

    map = result.astype(np.uint8)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    map = cv2.erode(map, kernel2)
    map = cv2.dilate(map, kernel1)
    # return information of roi
    result_contours = []
    result_contours_labels = []
    area_ratio = []
    img_h, img_w = image.shape[:2]
    for class_ind in range(1, 4):
        classmap = np.where(map == class_ind, 1, 0)
        # area_ratio.append(classmap.sum() / (img_h * img_w))
        classmap = classmap.astype(np.uint8)
        # contours,_ = cv2.findContours(classmap,cv2.RETR_LIST,2)
        contours, _ = cv2.findContours(classmap, cv2.RETR_EXTERNAL, 2)
        for contour in contours:
            result_contours.append(np.squeeze(contour, axis=(1,)).tolist())
            result_contours_labels.append(class_ind)

    # roi_dic = []
    # for (contour, label) in zip(result_contours, result_contours_labels):
    #     roi_dic.append([contour, label])

    return result_contours, result_contours_labels

@torch.no_grad()
def cal_bxr_np(image, C_net, mean, std, device):
    trans = transforms.Compose([transforms.ToTensor()])
    image = trans(image)
    for t, m, s in zip(image, mean, std):
        t.sub_(m).div_(s)
    image = image[None].cuda(device)
    outputs = C_net(image)

    points = outputs['pnt_coords'][0].cpu().numpy()
    scores = torch.softmax(outputs['cls_logits'][0], dim=-1).cpu().numpy()
    select_idx = np.where(np.logical_and(points[:,0]<image.shape[-1], points[:,1]<image.shape[-2]))[0]
    points = points[select_idx]
    scores = scores[select_idx]

    classes = np.argmax(scores, axis=-1)
    reserved_index = classes < 4
    points, classes = deduplicate(points[reserved_index], scores[reserved_index], 12)

    return points.astype(np.int), classes.astype(np.int)

def deduplicate(points, scores, interval):
    n = len(points)
    fused = np.full(n, False)
    result = np.zeros((0, 2))
    classes = np.array([])
    for i in range(n):
        if not fused[i]:
            fused_index = np.where(np.linalg.norm(points[[i]] - points[i:], 2, axis=1) < interval)[0] + i
            fused[fused_index] = True
            r_, c_ = np.where(scores[fused_index] == np.max(scores[fused_index]))
            p_ = np.max(scores[fused_index], axis=1)
            r_, c_, p_ = [r_[0]], [c_[0]], [p_[0]]
            result = np.append(result, points[fused_index[r_]], axis=0)
            classes = np.append(classes, c_)
    return result, classes

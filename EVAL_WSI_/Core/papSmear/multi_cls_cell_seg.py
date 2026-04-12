import numpy as np
import cv2
import torch
from torch.autograd import Variable
from skimage.feature import peak_local_max
from scipy.spatial.distance import cdist

def split_image(image, patch_size, overlap):
    h, w = image.shape[0:2]
    stride = patch_size - overlap
    patch_list = []
    num_y, num_x = 0, 0

    for y in range(0, h, stride):
        num_x = 0
        for x in range(0, w, stride):
            crop_img = image[y:y + patch_size, x:x + patch_size, :]
            crop_h, crop_w = crop_img.shape[0:2]
            pad_h, pad_w = patch_size - crop_h, patch_size - crop_w

            if pad_h > 0 or pad_w > 0:
                crop_img = np.pad(crop_img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')

            patch_list.append(crop_img)
            num_x += 1
        num_y += 1
    patch_image = np.array(patch_list)

    return patch_image, num_y, num_x


def reconstruct_mask(masks, patch_size, overlap, num_y, num_x):
    num_channel = masks.shape[1]
    stride = patch_size - overlap
    mask_h, mask_w = patch_size + (num_y - 1) * stride, patch_size + (num_x - 1) * stride
    result_mask = np.zeros((num_channel, mask_h, mask_w))
    mask_count = np.zeros((mask_h, mask_w, 1))
    for y in range(num_y):
        for x in range(num_x):
            i = y * num_x + x

            # print("x = {}, y = {}, i = {}".format(x, y, i))

            ys, ye = y * stride, y * stride + patch_size
            xs, xe = x * stride, x * stride + patch_size
            # import pdb; pdb.set_trace()
            result_mask[:, ys:ye, xs:xe] += masks[i]
            mask_count[ys:ye, xs:xe, :] += 1
    result_mask = result_mask.transpose(1, 2, 0)
    result_mask /= mask_count
    return result_mask

def generate_region_mask(image, net, patch_size=512, overlap=512, batch_size=4):
    image=cv2.resize(image,(512,512))
    img_h, img_w = image.shape[0:2]
    patch_imgs, num_y, num_x = split_image(image, patch_size, overlap)
    num_patches = patch_imgs.shape[0]
    patch_imgs = patch_imgs.transpose((0, 3, 1, 2))
    patch_imgs = patch_imgs * (2. / 255) - 1.

    results = []
    for i in range(0, num_patches, batch_size):
        # import pdb; pdb.set_trace()
        this_batch = patch_imgs[i:i + batch_size]
        with torch.no_grad():
            data_variable = Variable(torch.from_numpy(this_batch).float())
            if net.parameters().__next__().is_cuda:
                data_variable = data_variable.cuda(net.parameters().__next__().get_device())
                # result = net(data_variable)
                # import pdb; pdb.set_trace()
                result = net(data_variable)
                result = torch.softmax(result, dim=1)
                results.append(result.cpu().numpy())
                # results.append(net.forward(data_variable).cpu().data.numpy())

    results = np.concatenate(results)
    result_masks = reconstruct_mask(results, patch_size, overlap, num_y, num_x)
    result_masks = result_masks[0:img_h, 0:img_w, :]
    # result_masks=result_masks[:,:,0:-1]
    return result_masks

def generate_result_mask(image, net,region_mask, patch_size=512, overlap=512, batch_size=4):
    img_h, img_w = image.shape[0:2]
    patch_imgs, num_y, num_x = split_image(image, patch_size, overlap)
    num_patches = patch_imgs.shape[0]
    patch_imgs = patch_imgs.transpose((0, 3, 1, 2))
    patch_imgs = patch_imgs * (2. / 255) - 1.

    results = []
    for i in range(0, num_patches, batch_size):
        # import pdb; pdb.set_trace()
        this_batch = patch_imgs[i:i + batch_size]
        with torch.no_grad():
            data_variable = Variable(torch.from_numpy(this_batch).float())
            if net.parameters().__next__().is_cuda:
                data_variable = data_variable.cuda(net.parameters().__next__().get_device())
                # result = net(data_variable)
                # import pdb; pdb.set_trace()
                result = net(data_variable)
                result = torch.softmax(result, dim=1)
                results.append(result.cpu().numpy())
                # results.append(net.forward(data_variable).cpu().data.numpy())

    results = np.concatenate(results)
    result_masks = reconstruct_mask(results, patch_size, overlap, num_y, num_x)
    result_masks = result_masks[0:img_h, 0:img_w, :]
    result_masks=result_masks*region_mask
    # result_masks=result_masks[:,:,0:-1]
    return result_masks

def generate_result_mask_1(image, net, patch_size=512, overlap=512, batch_size=4):
    image = image.transpose((2, 0, 1))
    image = image * (2. / 255) - 1.

    results = []

    with torch.no_grad():
            data_variable = Variable(torch.from_numpy(image).float()).cuda()
            # if  net.parameters().__next__().is_cuda:
            # data_variable = data_variable.cuda()
            # result = net(data_variable)
            # import pdb; pdb.set_trace()
            result = net(data_variable)
            result = torch.sigmoid(result)
            results=result.cpu().numpy()
            # results.append(net.forward(data_variable).cpu().data.numpy())


    return results


def get_coordinate(voting_map, min_len=6):
    coordinates = peak_local_max(voting_map, min_distance=min_len, indices=True, exclude_border=min_len // 2)  # N by 2
    if coordinates.size == 0:
        coordinates = None  # np.asarray([])
        return coordinates

    boxes_list = [coordinates[:, 1:2], coordinates[:, 0:1], coordinates[:, 1:2], coordinates[:, 0:1]]
    coordinates = np.concatenate(boxes_list, axis=1)
    return coordinates


def post_process_mask(masks, threshold, resize_ratio):
    voting_map = np.max(masks, axis=2)
    voting_map[voting_map < threshold * 1.] = 0

    # import pdb; pdb.set_trace()
    # point_mask = (voting_map-np.min(voting_map))/(np.max(voting_map)-np.min(voting_map)) * 255
    bboxes = get_coordinate(voting_map, min_len=int(10 * resize_ratio))
    if bboxes is not None:
        x_coords = bboxes[:, 0]
        y_coords = bboxes[:, 1]
        center_point_1=bboxes[:, 0:2]
        center_point_2 = bboxes[:, 0:2]
        dist = np.array(cdist(center_point_1, center_point_2),dtype=float)
        dist[np.where(dist==0)]=100000
        label_map = np.argmax(masks, axis=2)
        label = label_map[y_coords, x_coords]
        #label[np.where(dist <= 10)[1]]=6
        return bboxes[:, 0:2], label
    else:
        bboxes = []
        label = []
        return bboxes,label


def cal_ki67_np_region(ori_img=None,R_net=None, resize_ratio=1):

    enhanced_img = ori_img
    region_results = generate_region_mask(enhanced_img, R_net, patch_size=256, overlap=128, batch_size=4)
    region_ind=np.argmax(region_results,axis=-1)
    region_mask=np.where(region_ind==5,1,0)
    region_mask=region_mask*255
    region_mask=np.asarray(region_mask,np.uint8)
    region_mask=cv2.resize(region_mask,(ori_img.shape[1],ori_img.shape[0]))
    region_mask=region_mask//255
    region_mask=region_mask[:,:,np.newaxis]

    return region_mask



def cal_ki67_np(ori_img=None, C_net=None, resize_ratio=1,a_mask=None):

    enhanced_img = ori_img

    result_masks = generate_result_mask(enhanced_img, C_net,a_mask, patch_size=512, overlap=128, batch_size=2)
    result_masks=result_masks[:,:,:-1]
    #result_masks_c = result_masks[:, :, 0:4]
    center_coords, labels = post_process_mask(masks=result_masks, threshold=0.3, resize_ratio=resize_ratio)

    return center_coords, labels
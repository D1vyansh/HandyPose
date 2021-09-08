import os
import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import utils.Mytransforms as Mytransforms


def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
  gridy, gridx = np.mgrid[0:size_h, 0:size_w]
  D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
  return np.exp(-D2 / 2.0 / sigma / sigma)

class cmu(data.Dataset):
  def __init__(self, root_dir, sigma, is_train, transform=None):
    self.num_joints = 21
    self.acc_idxs = [0, 1, 2, 3, 4, 5, 10, 11, 14, 15]
    self.edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], 
                  [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], 
                  [6, 8], [8, 9]]
    self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
    self.parts_num = 21
    self.stride = 8
    self.data_path = root_dir
    self.data_root = root_dir

    self.input_h, self.input_w   = 368, 368
    self.output_h, self.output_w = 46,46
    self.img_size = 256
    self.joints = 21
    self.is_train = is_train
    self.sigma = sigma
    #self.label_size = img_size // stride
    self.hm_gauss = 2

    img_names = json.load(open(os.path.join(self.data_root, 'partitions.json')))[is_train]
    annot = json.load(open(os.path.join(self.data_root, 'labels.json')))
    #print((annot[0]))
    #quit()
    self.num_samples = len(img_names)
    print('Loaded 2D {} {} samples'.format(is_train, self.num_samples))
    #self.aspect_ratio = 1.0 * self.input_w/ self.input_h   
    self.split = is_train
    self.annot = annot
    self.img_names = img_names
  
  def _load_image(self, index):
    path = '{}/imgs/{}'.format(
      self.data_path, self.img_names[index])
    image = cv2.imread(path)
    # print(path)
    return image
    
  def getBoundingBox(self, pts, height, width):
    x = []
    y = []

    for index in range(0, len(pts)):
        if float(pts[index][1]) >= 0 or float(pts[index][0]) >= 0:
            x.append(float(pts[index][0]))
            y.append(float(pts[index][1]))

    if len(x) == 0 or len(y) == 0:
        x_min = 0
        x_max = 0
        y_min = 0
        y_max = 0
    else:
        x_min = int(max(min(x), 0))
        x_max = int(min(max(x), width))
        y_min = int(max(min(y), 0))
        y_max = int(min(max(y), height))
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    center = torch.Tensor([center_x, center_y])
    return center

      
  def __getitem__(self, index):
    img_name = self.img_names[index]
    #print(img_name)
    img = self._load_image(index)
    points = torch.Tensor(self.annot[img_name])
    h, w, channel = np.shape(img)
    #print(img)
    center = self.getBoundingBox(points,h,w)
    scale = max(img.shape[0], img.shape[1]) * 1.0
    #print(center,scale)
    if center[0] != -1:
        center[1] = center[1] + 15 * scale
        scale = scale * 1.25
    nParts = points.size(0)
    kpt = points
    if img.shape[0] != 368 or img.shape[1] != 368:
        kpt[:, 0] = kpt[:, 0] * (368 / img.shape[1])
        kpt[:, 1] = kpt[:, 1] * (368 / img.shape[0])
        img = cv2.resize(img, (368, 368))
        height, width, _ = img.shape
    #cv2.imwrite('originalImage.png', img)
    #print(self.stride)
    heatmap = np.zeros((int(height / self.stride), int(width / self.stride), int(len(kpt))), dtype=np.float32)
    #print(np.shape(heatmap))
    for i in range(len(kpt)):
      # resize from 368 to 46
      x = int(kpt[i][0]) * 1.0 / self.stride
      y = int(kpt[i][1]) * 1.0 / self.stride
      heat_map = guassian_kernel(size_h=int(height / self.stride), size_w=int(width / self.stride), center_x=x,
                                 center_y=y, sigma=self.sigma)
      heat_map[heat_map > 1] = 1
      heat_map[heat_map < 0.0099] = 0
      heatmap[:, :, i] = heat_map

    #heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)
    #centermap = np.zeros((int(height / self.stride), int(width / self.stride), 1), dtype=np.float32)
    #center_map = guassian_kernel(size_h=int(height / self.stride), size_w=int(width / self.stride),
    #                             center_x=int(center[0] / self.stride), center_y=int(center[1] / self.stride), sigma=3)
    #center_map[center_map > 1] = 1
    #center_map[center_map < 0.0099] = 0
    #centermap[:, :, 0] = center_map
    img = Mytransforms.normalize(Mytransforms.to_tensor(img), [128.0, 128.0, 128.0],
                                 [256.0, 256.0, 256.0])
    heatmap = Mytransforms.to_tensor(heatmap)
    #centermap = Mytransforms.to_tensor(centermap)

    w = np.array(float(w))
    return img, heatmap, w
    
  def __len__(self):
    return self.num_samples



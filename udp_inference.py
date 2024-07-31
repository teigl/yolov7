
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import socket
import os
import configparser
import tensorflow as tf
from auto_pose.ae import factory as ae_factory
from auto_pose.ae import utils as ae_utils

class PoseEstimator:
  def __init__(self, opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf.compat.v1.disable_eager_execution()
    self.sess = tf.compat.v1.Session(config=tf_config)
    tf.compat.v1.keras.backend.set_session(self.sess)

    self.models = {}
    self.all_codebooks = {}
    self.all_train_args = {}
    self.all_pad_factors = {}
    self.all_patch_sizes = {}
    workspace_path = os.environ.get('AE_WORKSPACE_PATH')

    self.camera_matrix = np.eye(3)
    self.camera_matrix[0,0] = 807.4369608188626
    self.camera_matrix[1,1] = 806.4541893152723
    self.camera_matrix[0,2] = 646.0708692237737
    self.camera_matrix[1,2] = 353.68277427528596

    for model in opt.ae_models:
      label,name = model.split(':')
      label = int(label)

      log_dir = ae_utils.get_log_dir(workspace_path, name, opt.ae_group)
      train_cfg_file_path = ae_utils.get_train_config_exp_file_path(
        log_dir, name)
      ckpt_dir = ae_utils.get_checkpoint_dir(log_dir)

      train_args = configparser.ConfigParser()
      train_args.read(train_cfg_file_path) 

      self.models[label] = {}
      self.models[label]['train_args'] = train_args

      self.models[label] = train_args
      self.models[label]['pad_factor'] = train_args.getfloat('Dataset', 'PAD_FACTOR')
      self.models[label]['patch_size'] = (
        train_args.getint('Dataset', 'W'), train_args.getint('Dataset', 'H'))
      self.models[label]['codebook'] = ae_factory.build_codebook_from_name(
        model, opt.ae_group)
      saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=model))
      ae_factory.restore_checkpoint(self.sess, saver, ckpt_dir)

  def estimate(self, img, label, bb_xywh):
    crop = PoseEstimator.extract_square_patch(img, bb_xywh, 
      self.models[label]['pad_factor'], self.models[label]['patch_size'])
    Rs,ts,s = self.models[label]['codebook'].auto_pose6d(
      self.sess, crop, bb_xywh, self.camera_matrix, 1, 
      self.models[label]['train_args'], return_similarity=True)
    return Rs,ts,s

  def extract_square_pacth(img, bb_xywh, pad_factor, patch_size):
    x, y, w, h = np.array(bb_xywh).astype(np.int32)
    size = int(max(h, w)*pad_factor)

    left = max(x+w//2-size//2, 0)
    right = x+w//2+size//2
    top = max(y+h//2-size//2, 0)
    bottom = y+h//2+size//2

    # print(top, bottom, left, right)
    patch = img[top:bottom, left:right].copy()
    patch[:(y-top),:] = 0
    patch[(y+h-top):,:] = 0
    patch[:,:(x-left)] = 0
    patch[:,(x+w-left):] = 0
    patch = cv2.resize(
      patch, 
      patch_size, 
      interpolation=cv2.INTER_LINEAR)

    return patch


class Detector:
  def __init__(self, opt):
    # Initialize
    set_logging()
    self.device = select_device('0')

    # Load model
    self.model = attempt_load(opt.weights, map_location=self.device)  # load FP32 model
    stride = int(self.model.stride.max())  # model stride
    self.img_size = check_img_size(opt.img_size, s=stride)  # check img_size

    if opt.trace:
      self.model = TracedModel(self.model, self.device, opt.img_size)

    self.model.half()  # to FP16

    # Set Dataloader
    cudnn.benchmark = True  # set True to speed up constant image size inference

    # Get names and colors
    self.names = self.model.module.names if hasattr(self.model, 'module') \
      else self.model.names
    self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    self.view_img = opt.show_detections

    # Run inference
    self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(
      self.device).type_as(next(self.model.parameters())))  # run once
    self.old_img_shape = (1, -1, self.img_size, self.img_size)
  
  def detect(self, input_img):
    img = letterbox(input_img, self.img_size, stride=self.stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(self.device)
    img = img.half() #if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
      img = img.unsqueeze(0)

    # Warmup
    if self.old_img_shape != img.shape:
      print('got new image shape: ', img.shape)
      self.old_img_shape = img.shape
      for i in range(3):
        self.model(img, augment=opt.augment)[0]

    # Inference
    t1 = time_synchronized()
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
      pred = self.model(img, augment=opt.augment)[0]
    t2 = time_synchronized()

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, 
      classes=opt.classes, agnostic=opt.agnostic_nms)
    t3 = time_synchronized()

    # Process detections
    bb_xywh = np.zeros([])
    for i, det in enumerate(pred):  # detections per image
      if not len(det):
        continue
      s = ''
      gn = torch.tensor(input_img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
      # Rescale boxes from img_size to im0 size
      det[:,:4] = scale_coords(
        img.shape[2:], det[:, :4], input_img.shape).round()

      # Print results
      for c in det[:, -1].unique():
        n = (det[:, -1] == c).sum()  # detections per class
        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

      for *xyxy, conf, cls in reversed(det):
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

        if self.view_img:  # Add bbox to image
          label = f'{self.names[int(cls)]} {conf:.2f}'
          plot_one_box(xyxy, input_img, label=label, 
            color=self.colors[int(cls)], line_thickness=1)

      # Stream results
      if self.view_img:
        print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ' + \
          f'({(1E3 * (t3 - t2)):.1f}ms) NMS')
        cv2.imshow('YOLOv7', input_img)
        cv2.waitKey(1)  # 1 millisecond

def put_text(img, text, xpos, line):
  """
      wrapper for opencv put text
  """
  font_face = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 0.5
  font_color = (0,0,255)
  line_height = 30
  cv2.putText(img, text, (xpos,line*line_height), font_face, font_scale, font_color)

def run_server(opt, detector, pose_estimator):
  with socket.socket(family=socket.AF_INET,type=socket.SOCK_DGRAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('127.0.0.1',27060))

    t0 = time.time()
    input_image = np.zeros((256,512), dtype=np.uint8)
    t1 = time.time()

    while cv2.waitKey(1) == -1:
      try:
        sock.settimeout(1)
        data,addr = sock.recvfrom(1024)
      except socket.error:
        put_text(input_image, 'Timed out...', 200, 1)
        cv2.imshow('Detection', input_image)
        continue
      
      if opt.show_detections:
        input_image = cv2.imread(data.decode('utf-8'))
        image = input_image.copy()
        t0 = time.time()
      else:
        image = cv2.imread(data.decode('utf-8'))
      
      



      input_list, scale = preprocess(
        image, image_size, camera_matrix, translation_scale_norm)

      boxes, scores, labels, rotations, translations = \
        ep_model.predict_on_batch(input_list)

      boxes, scores, labels, rotations, translations = postprocess(
        boxes, scores, labels, rotations, translations, scale, score_threshold)

      similarities = np.zeros(len(rotations), dtype=np.float32)
      for i,(label,box) in enumerate(zip(labels,boxes)):
        if not (label in ae_data):
          continue
        bb_xywh = np.concatenate((box[:2], box[2:]-box[:2]))
        image_crop = extract_square_patch(image, bb_xywh, 
          ae_data[label]['pad_factor'], ae_data[label]['patch_size'])
        Rs,ts,s = ae_data[label]['codebook'].auto_pose6d(sess, image_crop, 
          bb_xywh, camera_matrix, 1, ae_data[label]['train_args'],
          return_similarity=True)
        
        rotations[i] = cv2.Rodrigues(Rs[0])[0].astype(np.float32).flatten()
        translations[i] = ts[0].astype(np.float32)
        similarities[i] = s

        if show_detections:
          cv2.rectangle(input_image, 
            box[:2].astype(np.int), box[2:].astype(np.int), 
            label_color(label))

      if show_detections:
        put_text(input_image, f'Detection: {int(1000*(time.time() - t0))}ms', 10, 1)
        draw_detections(input_image, boxes, scores, labels, rotations, translations,
          class_to_bbox_3D = label_to_3d_bboxes,
          camera_matrix = camera_matrix,
          label_to_name = label_to_name,
          draw_bbox_2d = draw_bbox_2d,
          draw_name = draw_name)

      similarities = 0.5*(similarities + 1)
      data = b''
      for i in range(rotations.shape[0]):
        data += labels[i].tobytes()
        data += rotations[i].tobytes()
        data += translations[i].tobytes()
        data += similarities[i].tobytes()

      if len(data) == 0:
        data = b'0'

      try:
        srv_socket.sendto(data, addr)
      except socket.error:
        pass
      
      if show_detections:
        put_text(input_image, f'Loop: {int(1000*(time.time() - t1))}ms', 10, 2)
        t1 = time.time()
        cv2.imshow('Detection', input_image)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
  parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
  parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
  parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
  parser.add_argument('--view-img', action='store_true', help='display results')
  parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
  parser.add_argument('--ae_group', type=str, default='wehak')
  parser.add_argument('--ae_models', nargs='*', type=str, default='0:dhandle')
  parser.add_argument('--ip', type=str, default='127.0.0.1')
  parser.add_argument('--port', type=int, default=27060)
  parser.add_argument('--show_detections', action='store_true')
  opt = parser.parse_args()
  print(opt)

  with torch.no_grad():
    detector = Detector(opt)

  pose_estimator = PoseEstimator(opt)

  run_server(opt, detector, pose_estimator)
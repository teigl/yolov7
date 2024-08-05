from utils.datasets import AutogenDataset
import cv2

objects = [{
  'class': 0,
  'path': 'test/bluelogic_handle.obj',
  # 'path': 'C:/Users/HakonT/ae_ws/mesh/dhandle.ply',
}]

images = ['test/test.jpg']

dataset = AutogenDataset(objects, images, 1)

while cv2.waitKey() != ord('q'):
  dataset.generate_batch(0, True)
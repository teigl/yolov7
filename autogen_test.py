from utils.datasets import AutogenDataset
import cv2

objects = [{
  'class': 0,
  'path': 'test/bluelogic_notext.obj',
  # 'path': 'test/cube.obj',
  # 'path': 'C:/Users/HakonT/ae_ws/mesh/dhandle.ply',
}]

images = 'test'

dataset = AutogenDataset(images, (640,640), 1, objects)

while cv2.waitKey() != ord('q'):
  dataset.generate_batch(0, True)
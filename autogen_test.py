from utils.datasets import AutogenDataset

objects = [{
  'class': 0,
  'path': 'test/bluelogic_handle.obj',
}]

images = ['test/2007_000175.jpg']

dataset = AutogenDataset(objects, images, 1)
dataset.generate_batch(0, True)
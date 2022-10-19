python train_aux.py ^
  --workers 8 ^
  --device 0 ^
  --batch-size 1 ^
  --data data/wehak.yaml ^
  --img 1280 1280 ^
  --cfg cfg/training/yolov7-w6-wehak.yaml ^
  --weights 'weights/yolov7-w6.pt' ^
  --name yolov7-w6-wehak ^
  --hyp data/hyp.scratch.custom.yaml

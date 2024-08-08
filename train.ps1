python train.py `
  --workers 0 `
  --device 0 `
  --batch-size 2 `
  --data data/subsea.yaml `
  --img 640 640 `
  --cfg cfg/training/yolov7.yaml `
  --weights 'weights/yolov7.pt' `
  --name yolov7-subsea `
  --hyp data/hyp.scratch.custom.yaml `
  --epochs 2 `

#!/bin/bash

python main.py validate     --config configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml     --data.init_args.path /data/cityscapes     --data.init_args.batch_size 1

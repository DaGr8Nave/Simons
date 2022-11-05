_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/CholecSeg8k_config.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
model = dict(
    backbone=dict(depth=18),
    decode_head=dict(c1_in_channels=64, c1_channels=12, in_channels=512, channels=128,num_classes=13),
    auxiliary_head=dict(in_channels=256, channels=64, num_classes=13),
    test_cfg=dict(mode='slide', crop_size=(480, 480), stride=(320, 320)))
optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)

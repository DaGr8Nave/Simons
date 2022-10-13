_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/CholecSeg8k_config.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

model = dict(
    decode_head=dict(num_classes=13),
    auxiliary_head=dict(num_classes=13),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(crop_size=(64, 64), stride=(42, 42))
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)

_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/CholecSeg8k_config.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type = 'BN', requires_grad=True)

model = dict(
    backbone=dict(
        embed_dims=64,
        num_layers=[3, 4, 18, 3]),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        num_classes=13,),
    test_cfg=dict(crop_size=(480, 480), stride=(1, 128)))

_base_ = ['./segformer_mit-b0_480x480_160k_cholecseg8k.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth'  # noqa

# model settings
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[2, 2, 2, 2]),
    decode_head=dict(in_channels=[64, 128, 320, 512], num_classes=13),
    test_cfg=dict(mode='slide', crop_size=(480,480), stride=(320,320)))

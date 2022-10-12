# Your dataset type defined in ./mmseg/datasets/__init__.py
dataset_type = 'CholecSeg8k'
# Correct path of your dataset
data_root = 'data/cholecseg8k'

img_norm_cfg = dict( # This img_norm_cfg is widely used because it is mean and std of ImageNet 1K pretrained model
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

#crop_size = (480, 480) # Crop size of image in training
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(320, 320)),
    #dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    #dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320, 320),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize'),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1, # Batch size of a single GPU
    workers_per_gpu=4, # Worker to pre-fetch data for each single GPU
    train=dict( # Train dataset config
        type=dataset_type, # Type of dataset, refer to mmseg/datasets/ for details.
        data_root=data_root, # The root of dataset.
        img_dir='img_dir/train', # The image directory of dataset.
        ann_dir='ann_dir/train',  # The annotation directory of dataset.
        pipeline=train_pipeline), # pipeline, this is passed by the train_pipeline created before.
    val=dict( # Validation dataset config.
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=test_pipeline), # Pipeline is passed by test_pipeline created before.
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=test_pipeline))

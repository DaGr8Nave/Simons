from .builder import DATASETS
from .custom import CustomDataset

# Register MyDataset class into DATASETS
@DATASETS.register_module()
class CholecSeg8k(CustomDataset):
    # Class names of your dataset annotations, i.e., actual names of corresponding label 0, 1, 2 in annotation segmentation maps
    CLASSES = ('Black Background', 'Abdominal Wall', 'Liver', 'G.I. Tract', 'Fat', 'Grasper', 'Connective Tissue', 'Blood', 'Cystic Duct', 'L-hook Electrocautery', 'Gallbladder', 'Hepatic Vein', 'Liver Ligament')

    # BGR value of corresponding classes, which are used for visualization
    PALETTE = [[127,127,127], [210, 140, 140], [255, 114, 114], [231, 70, 156], [186, 183, 75], [170, 255, 0], [255, 85, 0], [255, 0, 0], [255, 255, 0], [169, 255, 184], [255, 160, 165], [0, 50, 128], [111, 74, 0]]

    # The formats of image and segmentation map are both .png in this case
    def __init__(self, **kwargs):
        super(CholecSeg8k, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False, # reduce_zero_label is False because label 0 is background (first one in CLASSES above)
            **kwargs)
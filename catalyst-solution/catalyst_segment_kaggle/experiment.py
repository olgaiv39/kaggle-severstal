from collections import OrderedDict

from catalyst.dl.experiment import ConfigExperiment
import albumentations as A

from mlcomp.contrib.dataset.segment import ImageWithMaskDataset
from mlcomp.contrib.transform.albumentations import ChannelTranspose
from mlcomp.utils.config import parse_albu


class Experiment(ConfigExperiment):
    @staticmethod
    def prepare_train_transforms(aug=None):
        transforms = [
        
         A.HorizontalFlip(p=0.6),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        #A.PadIfNeeded(min_height=960, min_width=960, always_apply=False, border_mode=0),
        A.RandomCrop(height=256, width=256, always_apply=False),

        A.IAAAdditiveGaussianNoise(p=0.3),
        A.IAAPerspective(p=0.4),

        A.OneOf(
            [
            A.CLAHE(p=1),
            A.RandomBrightness(p=1),
            A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
            A.IAASharpen(p=1),
            A.Blur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.8,
        )
            
        ]

        if aug:
            transforms += parse_albu(aug)

        transforms.extend(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ChannelTranspose()
            ]
        )
        return A.Compose(transforms)

    @staticmethod
    def prepare_test_transforms():
        transforms = [
            A.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
            ChannelTranspose()
        ]
        return A.Compose(transforms)

    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()

        params = self.stages_config[stage]['data_params']
        common = {
            'img_folder': params['img_folder'],
            'mask_folder': params['mask_folder'],
            'fold_csv': params['fold_csv'],
            'fold_number': params['fold_number'],
            'num_classes': 4,
            'max_count': params.get('max_count', None)
        }

        train = ImageWithMaskDataset(
            **common,
            is_test=False,
            transforms=Experiment.prepare_train_transforms(params.get('aug')),
            crop_positive=(256, 256, 1)
        )
        valid = ImageWithMaskDataset(
            **common,
            is_test=True,
            transforms=Experiment.prepare_test_transforms()
        )

        datasets['train'] = train
        datasets['valid'] = valid
        return datasets

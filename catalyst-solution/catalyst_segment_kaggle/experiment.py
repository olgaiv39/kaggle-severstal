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
            A.HorizontalFlip()
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

from collections import OrderedDict
from copy import deepcopy

import cv2

import numpy as np
from tqdm import tqdm

from mlcomp.contrib.dataset.segment import ImageWithMaskDataset
from mlcomp.contrib.metrics.dice import dice_numpy
from mlcomp.worker.executors import Executor
from mlcomp.worker.executors.valid import Valid
from mlcomp.worker.reports.segmenation import SegmentationReportBuilder

from catalyst_segment import Experiment
from utils.executor_mixin import ExecutorMixin


@Executor.register
class ValidSeverstal(Valid, ExecutorMixin):
    def __init__(self, **kwargs):
        cache_names = ['y', 'y_segment']
        super().__init__(
            layout='img_segment', cache_names=cache_names, **kwargs
        )

        self.x_source = ImageWithMaskDataset(
            img_folder='data/input/train_images',
            mask_folder='data/train_masks',
            fold_csv='data/masks.csv',
            fold_number=self.fold_number,
            is_test=True,
            transforms=Experiment.prepare_test_transforms(),
            num_classes=4,
            max_count=self.max_count
        )
        self.builder = None
        self.x = None
        self.dices = []

    def create_base(self):
        colors = [
            (255, 255, 0),  # yellow
            (0, 155, 191),  # light blue
            (148, 0, 211),  # purple
            (255, 0, 0)  # red
        ]

        self.builder = SegmentationReportBuilder(
            session=self.session,
            layout=self.layout,
            task=self.task,
            name=self.name,
            colors=colors,
            plot_count=self.plot_count
        ) if self.layout else None

    def adjust_part(self, part):
        self.x = deepcopy(self.x_source)
        self.x.data = self.x.data[part[0]:part[1]]

    def count(self):
        return len(self.x_source)

    def score(self, preds):
        dices_by_photo = []

        for i, img_pred in tqdm(enumerate(preds)):
            dice_by_photo = []

            for j, c in enumerate(img_pred):
                target = self.x[i]['targets'][j]
                score = dice_numpy(target, c, empty_one=True)
                self.dices.append(score)
                dice_by_photo.append(score)

            dices_by_photo.append(np.mean(dice_by_photo))

        return {'dice': dices_by_photo}

    def score_final(self):
        return np.mean(self.dices)

    def _plot_main(self, preds, scores):
        imgs = []
        attrs = []
        targets = []

        self.x.transforms = None

        for i, img_pred in tqdm(enumerate(preds)):
            if i >= self.builder.plot_count:
                break

            row = self.x[i]
            attr = {}

            for j, c in enumerate(img_pred):
                contours, _ = cv2.findContours(
                    c, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
                )
                attr[f'attr{i + 1}'] = len(contours)

            imgs.append(row['features'])
            targets.append(row['targets'].astype(np.uint8))
            attr['attr5'] = sum(attr.values())
            attrs.append(attr)

        preds = OrderedDict({'y': preds})
        if 'y_segment' in self.cache:
            preds['y_segment'] = self.cache['y_segment']

        self.builder.process_pred(
            imgs=imgs,
            preds=preds,
            targets=targets,
            attrs=attrs,
            scores=scores
        )

    def plot(self, preds, scores):
        self._plot_main(preds, scores)

    def plot_final(self, score):
        self.builder.process_scores({'dice': score})

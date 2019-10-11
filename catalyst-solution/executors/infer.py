import pickle
from collections import OrderedDict
from copy import deepcopy

import cv2

from mlcomp.contrib.dataset.classify import ImageDataset

from mlcomp.worker.executors import Executor
from catalyst_segment.experiment import Experiment
from mlcomp.worker.executors.infer import Infer
from mlcomp.worker.reports.segmenation import SegmentationReportBuilder
from utils.executor_mixin import ExecutorMixin


@Executor.register
class InferSeverstal(Infer, ExecutorMixin):
    def __init__(self, **kwargs):
        cache_names = ['y', 'y_segment']
        super().__init__(
            layout='img_segment', cache_names=cache_names, **kwargs
        )

        if self.test:
            self.x_source = ImageDataset(
                img_folder='data/input/test_images',
                is_test=True,
                transforms=Experiment.prepare_test_transforms(),
                max_count=self.max_count
            )
        else:
            self.x_source = ImageDataset(
                img_folder='data/input/train_images',
                fold_csv='data/masks.csv',
                fold_number=0,
                is_test=True,
                max_count=self.max_count,
                transforms=Experiment.prepare_test_transforms()
            )

        self.builder = None
        self.x = None
        self.res = []

    def count(self):
        return len(self.x_source)

    def adjust_part(self, part):
        self.x = deepcopy(self.x_source)
        self.x.data = self.x.data[part[0]:part[1]]

    def create_base(self):
        colors = [
            (0, 255, 255),  # yellow
            (155, 191, 0),  # light blue
            (211, 0, 148),  # purple
            (0, 0, 255)  # red
        ]

        self.builder = SegmentationReportBuilder(
            session=self.session,
            layout=self.layout,
            task=self.task,
            name=self.name,
            colors=colors,
            plot_count=self.plot_count
        )

    def save(self, preds, folder: str):
        for p in preds:
            img_channels = []
            for c in p:
                retval, buffer = cv2.imencode('.png', (c + 6) * 18.2)
                img_channels.append(buffer)
            self.res.append(img_channels)

    def save_final(self, folder):
        path = f'{folder}/{self.model_name}_{self.suffix}'
        pickle.dump(self.res, open(path, 'wb'))

    def _plot_main(self, preds):
        imgs = []
        attrs = []

        self.x.transforms = None

        for i, row in enumerate(self.x):
            if i >= self.builder.plot_count:
                break

            imgs.append(row['features'])

            attr = {}
            attrs.append(attr)

        preds = OrderedDict({'y': preds})
        if 'y_segment' in self.cache:
            preds['y_segment'] = self.cache['y_segment']

        self.builder.process_pred(imgs=imgs, preds=preds, attrs=attrs)

    def plot(self, preds):
        self._plot_main(preds)

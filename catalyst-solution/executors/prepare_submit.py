import os
from collections import OrderedDict
from copy import deepcopy

import pandas as pd

from mlcomp.contrib.dataset.classify import ImageDataset
from mlcomp.contrib.transform.rle import mask2rle

from mlcomp.worker.executors import Executor
from catalyst_segment.experiment import Experiment
from mlcomp.worker.executors.prepare_submit import PrepareSubmit
from mlcomp.worker.reports.segmenation import SegmentationReportBuilder
from utils.executor_mixin import ExecutorMixin


@Executor.register
class PrepareSubmitSeverstal(PrepareSubmit, ExecutorMixin):
    def __init__(self, **kwargs):
        cache_names = ['y', 'y_segment']
        super().__init__(
            layout='img_segment', cache_names=cache_names, **kwargs
        )

        self.x_source = ImageDataset(
            img_folder='data/input/test_images',
            is_test=True,
            transforms=Experiment.prepare_test_transforms(),
            max_count=self.max_count
        )

        self.builder = None
        self.x = None
        self.submit_res = []

    def count(self):
        return len(self.x_source)

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

    def adjust_part(self, part):
        self.x = deepcopy(self.x_source)
        self.x.data = self.x.data[part[0]:part[1]]

    def submit(self, preds):
        for i in range(len(self.x)):
            file = os.path.basename(self.x[i]['image_file'])
            for j, c in enumerate(preds[i]):
                encoded_pixels = mask2rle(c)
                im_cl = f'{file}_{j + 1}'

                self.submit_res.append(
                    {
                        'ImageId_ClassId': im_cl,
                        'EncodedPixels': encoded_pixels
                    }
                )

    def submit_final(self, folder):
        df = pd.DataFrame(self.submit_res
                          )[['ImageId_ClassId', 'EncodedPixels']]
        df.to_csv(f'{folder}/{self.model_name}_{self.suffix}.csv', index=False)

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

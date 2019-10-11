import pickle
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from mlcomp.contrib.dataset.classify import ImageDataset

from mlcomp.worker.executors import Executor

from mlcomp.worker.executors.infer import Infer
from mlcomp.worker.reports.classification import ClassificationReportBuilder

from catalyst_classify.experiment import Experiment
from utils.executor_mixin import ExecutorMixin


@Executor.register
class InferClassifySeverstal(Infer, ExecutorMixin):
    def __init__(self, **kwargs):
        cache_names = ['y']
        super().__init__(cache_names=cache_names, layout='img_classify',
                         **kwargs)

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

    def create_base(self):
        self.builder = ClassificationReportBuilder(
            session=self.session,
            task=self.task,
            layout=self.layout,
            name=self.name,
            plot_count=self.plot_count
        )
        self.builder.create_base()

    def count(self):
        return len(self.x_source)

    def adjust_part(self, part):
        self.x = deepcopy(self.x_source)
        self.x.data = self.x.data[part[0]:part[1]]

    def save(self, preds, folder: str):
        self.res.extend(preds)

    def save_final(self, folder):
        pickle.dump(np.array(self.res),
                    open(f'{folder}/{self.model_name}_{self.suffix}.p', 'wb'))

    def _plot_main(self, preds):
        imgs = []
        attrs = []

        self.x.transforms = None

        for i, img_pred in tqdm(enumerate(preds)):
            if i >= self.builder.plot_count:
                break

            row = self.x[i]
            attr = {'attr1': img_pred.argmax()}

            imgs.append(row['features'])
            attrs.append(attr)

        self.builder.process_pred(
            imgs=imgs,
            preds=preds,
            attrs=attrs
        )

    def plot(self, preds):
        self._plot_main(preds)

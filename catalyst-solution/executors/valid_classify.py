from copy import deepcopy

import numpy as np
from mlcomp.worker.reports.classification import ClassificationReportBuilder
from tqdm import tqdm

from mlcomp.contrib.dataset.segment import ImageWithMaskDataset
from mlcomp.worker.executors import Executor
from mlcomp.worker.executors.valid import Valid

from catalyst_segment import Experiment
from utils.executor_mixin import ExecutorMixin


@Executor.register
class ValidClassifySeverstal(Valid, ExecutorMixin):
    def __init__(self, **kwargs):
        super().__init__(
            layout='img_classify', **kwargs
        )

        self.x_source = ImageWithMaskDataset(
            img_folder='data/input/train_images',
            mask_folder='data/train_masks',
            fold_csv='data/masks.csv',
            fold_number=self.fold_number,
            is_test=True,
            transforms=Experiment.prepare_test_transforms(),
            num_classes=4,
            max_count=self.max_count,
            include_binary=True
        )
        self.builder = None
        self.x = None
        self.scores = []

    def create_base(self):
        self.builder = ClassificationReportBuilder(
            session=self.session,
            layout=self.layout,
            task=self.task,
            name=self.name,
            plot_count=self.plot_count
        ) if self.layout else None

    def adjust_part(self, part):
        self.x = deepcopy(self.x_source)
        self.x.data = self.x.data[part[0]:part[1]]

    def count(self):
        return len(self.x_source)

    def score(self, preds):
        res = []

        for i, img_pred in tqdm(enumerate(preds)):
            score = img_pred.argmax() == self.x[i]['empty_all']
            res.append(score)

        return {'accuracy': res}

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
            attr = {'attr1': img_pred.argmax(), 'attr2': row['empty_all']}

            imgs.append(row['features'])
            targets.append(row['empty_all'])
            attrs.append(attr)

        self.builder.process_pred(
            imgs=imgs,
            preds=preds,
            targets=targets,
            attrs=attrs,
            scores={'accuracy': scores}
        )

    def plot(self, preds, scores):
        self._plot_main(preds, scores)

    def plot_final(self, score):
        self.builder.process_scores({'accuracy': score})

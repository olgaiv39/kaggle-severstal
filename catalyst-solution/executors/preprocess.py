import pandas as pd

from mlcomp.contrib.split import stratified_group_k_fold
from mlcomp.worker.executors import Executor


@Executor.register
class Preprocess(Executor):
    def work(self):
        df = pd.read_csv('data/input/train.csv')
        df['exists'] = df['EncodedPixels'].notnull().astype(int)

        df['image_name'] = df['ImageId_ClassId'].map(
            lambda x: x.split('_')[0].strip()
        )
        df['class_id'] = df['ImageId_ClassId'].map(
            lambda x: int(x.split('_')[-1])
        )
        df['class_id'] = [
            row.class_id if row.exists else 0 for row in df.itertuples()
        ]
        df['fold'] = stratified_group_k_fold(
            label='class_id', group_column='image_name', df=df, n_splits=5
        )
        df.to_csv('data/fold.csv', index=False)


if __name__ == '__main__':
    Preprocess().work()
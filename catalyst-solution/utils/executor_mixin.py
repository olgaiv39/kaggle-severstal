from mlcomp.worker.executors.base.equation import Equation

from utils.postprocess import PostProcessMixin


class ExecutorMixin(Equation, PostProcessMixin):
    def load(self, file: str = None):
        return super().load(file=file) / 18.2 - 6
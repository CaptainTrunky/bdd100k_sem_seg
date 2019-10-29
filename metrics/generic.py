from functools import lru_cache
import numpy as np


class Metric:
    def __init__(self):
        self.steps = list()

    @staticmethod
    def _check_tensors(a, b):
        if len(a.shape) != 4 or len(b.shape) != 4:
            raise RuntimeError(f'Expecting 4D tensors BxCxHxW, got a = {a.shape}; b = {b.shape}')

        if a.shape != b.shape:
            raise Runtime(f'Shapes must be equal, got , got a = {a.shape}; b = {b.shape}')

    @staticmethod
    def _true_positive(a, b, label):
        tp_pred = (a == label).astype(bool)
        tp_label = (b == label).astype(bool)

        return np.logical_and(tp_pred, tp_label)

    @staticmethod
    def _false_positive(a, b, label):
        tp_pred = (a == label).astype(bool)
        tp_label = (b != label).astype(bool)

        return np.logical_and(tp_pred, tp_label)

    @staticmethod
    def _false_negative(a, b, label):
        tp_pred = (a != label).astype(bool)
        tp_label = (b == label).astype(bool)
 
        return np.logical_and(tp_pred, tp_label)

    def compute(self, predict, labels, label):
        raise NotImplementedError()

    def update(self, predict, labels, label):
        val = self.compute(predict, labels, label)

        self.steps.append(val)


class Precision(Metric):
    def __init__(self):
        super().__init__()

    @staticmethod
    def compute(predict, labels, label):
        Metric._check_tensors(predict, labels)

        tp = Metric._true_positive(predict, labels, label).sum().astype(np.float32)
        fp = Metric._false_positive(predict, labels, label).sum().astype(np.float32)

        return 1.0 if np.isclose(tp + fp, 0.0) else tp / (tp + fp)


class Recall(Metric):
    def __init__(self):
        super().__init__()

    @staticmethod
    def compute(predict, labels, label):
        Metric._check_tensors(predict, labels)

        tp = Metric._true_positive(predict, labels, label).sum().astype(np.float32)
        fn = Metric._false_negative(predict, labels, label).sum().astype(np.float32)
     
        return 1.0 if np.isclose(tp + fn, 0.0) else tp / (tp + fn)


class F1Score(Metric):
    def __init__(self):
        super().__init__()

    @staticmethod
    def compute(predict, labels, label):
        Metric._check_tensors(predict, labels)

        prec = Precision.compute(predict, labels, label)
        recall = Recall.compute(predict, labels, label)

        if np.isclose(prec + recall, 0.0):
            score = 0.0
        else:
            score = 2 * prec * recall / (prec + recall)

        return score


class IoUMetric(Metric):
    def __init__(self):
        super().__init__()

    def compute(self, predict, labels, label):
        Metric._check_tensors(predict, labels)

        mask_predict = (predict == label).astype(bool)
        mask_label = (labels == label).astype(bool)

        intersection = np.logical_and(mask_predict, mask_label).sum().astype(np.float32)
        union = np.logical_or(mask_predict, mask_label).sum().astype(np.float32)

        if np.isclose(intersection, 0.0):
            score = 0.0
        else:
            score = intersection / union

        return score


@lru_cache(maxsize=1)
def get_all_metrics():
    return {
        'f1_score': F1Score(),
        'precision': Precision(),
        'recall' : Recall(),
        'iou': IoUMetric()
    }


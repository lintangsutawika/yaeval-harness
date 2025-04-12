import numpy as np

def classical_pass_at_k(prediction, ground_truth, k=-1, metric_fn=None):

    if metric_fn is None:
        metric_fn = lambda x, y: x == y

    if k == -1:
        k = len(prediction)

    for pred in prediction:
        if metric_fn(pred, ground_truth):
            return 1
    return 0

def openai_pass_at_k(prediction, ground_truth, k=-1, metric_fn=None):

    def _pass_at_k(n, c, k):
        """
        :param n: total number of samples
        :param c: number of correct samples
        :param k: k in pass@$k$
        """
        if n - c < k: return 1.0
        return 1.0 - np.prod(1.0-k/np.arange(n-c+1,n+1))


    if metric_fn is None:
        metric_fn = lambda x, y: x == y

    if k == -1:
        k = len(prediction)

    num_correct = 0
    for pred in prediction:
        num_correct += int(metric_fn(pred, ground_truth))

    return _pass_at_k(len(prediction), num_correct, k)
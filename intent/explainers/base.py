"""
TODO
    - How do we pick a comparable way to choose train instances for explaining a single test instance.
        * e.g. TracIn measures the test loss; Trex measures which instances have the biggest imapce on the prediction.
"""
from abc import abstractmethod

from .parsers import parse_model


class Explainer(object):
    """
    Abstract class that all explainers must implement.
    """
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, model, X, y):
        """
        - Convert model to internal standardized tree structures.
        - Perform any initialization necessary for the chosen method.

        Input
            model: tree ensemble.
            X: training data.
            y: training targets.
        """
        self.model_ = parse_model(model)
        assert self.model_.tree_type in ['rf', 'gbdt']
        assert self.model_.task_ in ['regression', 'binary', 'multiclass']

    @abstractmethod
    def get_self_influence(self):
        """
        - Compute influence of each training instance on itself.
        - Provides a global perspective of which training intances
          are most important.

        Return
            - Regression and binary: 1d array of shape=(no. train).
            - Multiclass: 2d array of shape=(no. train, no. classes).
            - Array is returned in the same order as the traing data.
        """
        pass

    @abstractmethod
    def explain(self, X, y):
        """
        - Compute influence of each training instance on the test instance loss.
        - Provides a local explanation of the test instace prediction.

        Input
            - X: 2d array of shape=(1, no. feature).
            - y: 1d array of shape=(target,). True or predicted label (see notes).

        Return
            - Regression and binary: 1d array of shape=(no. train).
            - Multiclass: 2d array of shape=(no. train, no. classes).
            - Array is returned in the same order as the traing data.

        Notes:
            - To measure test loss, pass the true label of y.
            - To measure test loss in terms of the predicted label,
              pass the predicted label of y.
        """
        pass

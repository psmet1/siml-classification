import numpy as np

class BinaryClassificationSimulation:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def simulate(self, true_class, TPR, FPR):
        '''
        Simulate the prediction of a binary classifier.
        
        Parameters:
            true_class (str): The true value of the instance ('0' or '1').
            TPR (float): True Positive Rate.
            FPR (float): False Positive Rate.
        
        Returns:
            str: The predicted class ('0' or '1').
        '''
        r = np.random.random()
        if true_class == 0:
            return 0 if r <= TPR else 1
        else:
            return 1 if r <= (1 - FPR) else 0

class MultiClassClassificationSimulation:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def simulate(self, true_class, TPR, FPR):
        '''
        Simulate the prediction of a multiclass classifier.
        
        Parameters:
            true_class (str): The true class of the instance.
            TPR (dict): True Positive Rates for each class (e.g., {'A': 0.8, 'B': 0.7, 'C': 0.9}).
            FPR (dict): False Positive Rates for each class (e.g., {'A': 0.1, 'B': 0.2, 'C': 0.15}).
        
        Returns:
            str: The predicted class.
        '''
        all_classes = list(TPR.keys())
        r = np.random.random()
        if r <= TPR[true_class]:
            return true_class
        other_classes = [c for c in all_classes if c != true_class]
        total_FPR = sum(FPR[c] for c in other_classes)
        normalized_FPR = {c: FPR[c] / total_FPR for c in other_classes}
        r2 = np.random.random()
        cumulative_prob = 0
        for c in other_classes:
            cumulative_prob += normalized_FPR[c]
            if r2 <= cumulative_prob:
                return c
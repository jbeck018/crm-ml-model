import numpy as np
from numba import jit, prange

# ... existing imports ...

@jit(nopython=True, parallel=True)
def calculate_metrics(data):
    # ... existing code ...

@jit(nopython=True)
def calculate_single_metric(metric_func, data):
    # ... existing code ...

class Analysis:
    # ... existing code ...

    @staticmethod
    @jit(nopython=True, parallel=True)
    def perform_analysis(data, metrics):
        results = {}
        for metric in prange(len(metrics)):
            metric_name = metrics[metric]
            metric_func = getattr(np, metric_name)
            results[metric_name] = calculate_single_metric(metric_func, data)
        return results

    # ... rest of the class ...
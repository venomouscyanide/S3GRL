import time
from ray.tune import Stopper


class TimeStopper(Stopper):
    def __init__(self):
        self._start = time.time()
        self._deadline = 60 * 30  # 30 minutes max run across all experiments

    def __call__(self, trial_id, result):
        return False

    def stop_all(self):
        return time.time() - self._start > self._deadline

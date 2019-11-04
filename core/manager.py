from pathlib import Path
import logging


class ExperimentManager:
    def __init__(self, config, run):
        self._run = run

        self._root_dir = Path(config.experiments_dir)
        if not self._root_dir.exists():
            logging.warning(f'Creating {self._root_dir.as_posix()} directory')
            self._root_dir.mkdir()
            
        self._experiment_dir = self._root_dir / str(self._run._id)
 
        if self._experiment_dir.exists():
            raise RuntimeError(f'{self._experiment_dir.as_posix()} already exists')

        self._experiment_dir.mkdir()

        self._artifacts_dir = self._experiment_dir / 'images'
        self.artifacts.mkdir()

        self._weights_dir = self._experiment_dir / 'weights'
        self.weights.mkdir()

    @property
    def run(self):
        return self._run

    @property
    def exp_dir(self):
        return self._experiment_dir

    @property
    def weights(self):
        return self._weights_dir

    @property
    def artifacts(self):
        return self._artifacts_dir


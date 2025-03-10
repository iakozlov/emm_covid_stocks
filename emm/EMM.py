import sys
import logging
import pandas as pd

from copy import deepcopy
from typing import Any, List, Optional, Union
from multiprocessing import Manager, Process, Queue, cpu_count

from beam import Beam
from util import downsize
from subgroup import Subgroup
from evaluation_metrics import metrics, cleanup
from visualization import visualizations
from description import Description
from workers import create_subgroups, evaluate_subgroups, beam_adder


# logger = logging.getLogger(__name__)
m = Manager()
evaluate_queue = m.Queue()
add_queue = m.Queue()


class EMM:

    def __init__(self, width: int, depth: int, evaluation_metric: Union[str, callable], n_jobs: int = -1,
                 strategy: str = 'maximize', n_bins: int = 10, bin_strategy: Optional[str] = 'equidepth',
                 candidate_size: int = None, log_level=50):
        logging.basicConfig(filename=None, level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
        self.depth = depth
        self.evaluation_metric = evaluation_metric
        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = min(n_jobs, cpu_count())
        if hasattr(evaluation_metric, '__call__'):
            self.evaluation_function = evaluation_metric
        else:
            try:
                self.evaluation_function = metrics[evaluation_metric]
            except KeyError:
                raise ValueError(f"Nu such metric: {evaluation_metric}")
        self.settings = dict(
            strategy=strategy,
            width=width,
            n_bins=n_bins,
            bin_strategy=bin_strategy,
            candidate_size=candidate_size
        )
        self.beam = None
        self.target_columns = None
        self.dataset_target = None
        self.dataset = None

    def __setattr__(self, name, value):
        if name in ['depth', 'n_jobs'] and not isinstance(value, int):
            raise TypeError(f"Invalid type for setting {name}: only integers are allowed")
        if name == 'settings':
            if not isinstance(value['width'], int):
                raise TypeError("Invalid type for setting width: only integers are allowed")
            if value['strategy'] not in ['minimize', 'maximize']:
                raise ValueError(
                    f"Invalid option for strategy: {value['strategy']}, allowed options are `maximize` and `minimize`")
            if not isinstance(value['n_bins'], int):
                raise TypeError(f"Invalid type for setting {value['n_bins']}: only integers are allowed")
            if not isinstance(value['candidate_size'], int) and value['candidate_size'] is not None:
                raise TypeError(f"Invalid type for setting {value['n_bins']}: only integers or None are allowed")
            if not value['bin_strategy'] in ['equiwidth', 'equidepth', None]:
                raise ValueError(f"Invalid option for bin_strategy: {value['bin_strategy']}, allowed options are "
                                 f"`equidepth`. `equiwidth` and None")
        super().__setattr__(name, value)

    def search(self, data: pd.DataFrame, target_cols: Union[List[str], str], descriptive_cols: List[str] = None):
        logging.info("Start")
        df = deepcopy(data)
        data, translations = downsize(deepcopy(data))
        self.settings['object_cols'] = translations
        self.dataset = Subgroup(df, Description('all'))
        _, self.dataset.target = self.evaluation_function(df[target_cols], df[target_cols])
        self.beam = Beam(self.dataset, self.settings)
        target_cols = list(target_cols,)
        if descriptive_cols is None:
            descriptive_cols = [c for c in data.columns if c not in target_cols]
        elif any(c in descriptive_cols for c in target_cols):
            raise ValueError("The target and descriptive columns may not overlap!")
        if any(c not in data.columns for c in descriptive_cols + target_cols):
            raise ValueError("All specified columns should be present in the dataset")
        self.dataset_target = df[target_cols]
        self.target_columns = target_cols
        while self.depth > 0:
            self.make_subgroups(descriptive_cols)
            self.depth -= 1
            self.beam.select_cover_based()
        #print(translations)
        #self.beam.decrypt_descriptions(translations)
        #self.beam.print()
        for subgroup in self.beam.subgroups:
            subgroup.print()
        cleanup()

    def visualise(self, vis_type: Union[callable, str] = None, subgroups: int = None, cols: int = 3,
                  include_dataset=True, target_columns = None):
        if vis_type is None:
            vis_type = self.evaluation_metric
        if subgroups is None:
            subgroups = len(self.beam.subgroups)
        if hasattr(vis_type, '__call__'):
            vis_type(self.dataset, self.beam.subgroups, target_columns, self.settings['object_cols'], cols,
                     subgroups, include_dataset)
        else:
            try:
                visualizations[vis_type](self.dataset, self.beam.subgroups, target_columns,
                                         self.settings['object_cols'], cols, subgroups, include_dataset)
            except KeyError as e:
                raise ValueError(f"Nu such visualization: {vis_type}")

    def make_subgroups(self, cols: List[str]):
        beam_workers = []
        for _ in range(self.n_jobs):
            w = Process(target=evaluate_subgroups, args=(evaluate_queue, add_queue, self.target_columns,
                                                         self.dataset_target, self.evaluation_function))
            w.start()
            beam_workers.append(w)
        for subgroup in self.beam.subgroups:
            for col in cols:
                create_subgroups(subgroup, col, evaluate_queue, self.settings)
        for _ in range(self.n_jobs):
            evaluate_queue.put('done')
        beam_adder(add_queue, self.beam, self.n_jobs)
        for w in beam_workers:
            w.join()
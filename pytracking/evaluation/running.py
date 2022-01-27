import numpy as np
import multiprocessing
import os
from itertools import product
from pytracking.evaluation import Sequence, Tracker


def run_sequence(seq: Sequence, tracker: Tracker, debug=False):
    """Runs a tracker on a sequence."""

    base_results_path = '{}/{}'.format(tracker.results_dir, seq.name)
    if not os.path.isdir(base_results_path):
        os.mkdir(base_results_path)
    results_path = '{}/{}.txt'.format(base_results_path, seq.name)
    times_path = '{}/{}_time.value'.format(base_results_path, seq.name)
    conf_path = '{}/{}_confidence.value'.format(base_results_path, seq.name)

    if os.path.isfile(results_path) and not debug:
        return

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

    if debug:
        tracked_bb, exec_times, tracked_conf = tracker.run(seq, debug=debug)
    else:
        try:
            tracked_bb, exec_times, tracked_conf = tracker.run(seq, debug=debug)
        except Exception as e:
            print(e)
            return

    tracked_bb = np.array(tracked_bb).astype(float)
    exec_times = np.array(exec_times).astype(float)
    tracked_conf = np.array(tracked_conf).astype(float)

    print('FPS: {}'.format(len(exec_times) / exec_times.sum()))
    if not debug:
        np.savetxt(results_path, tracked_bb, delimiter=',', fmt='%f')
        np.savetxt(times_path, exec_times, delimiter='\t', fmt='%f')
        np.savetxt(conf_path, tracked_conf, delimiter='\t', fmt='%f')


def run_dataset(dataset, trackers, debug=False, threads=0):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
    """
    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        for seq in dataset:
            for tracker_info in trackers:
                run_sequence(seq, tracker_info, debug=debug)
    elif mode == 'parallel':
        param_list = [(seq, tracker_info, debug) for seq, tracker_info in product(dataset, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    print('Done')

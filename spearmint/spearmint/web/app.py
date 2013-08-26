import os
import sys
import json

from flask import Flask, render_template, redirect, url_for

from spearmint.ExperimentGrid import ExperimentGrid
from spearmint.helpers import load_experiment
from spearmint.spearmint_pb2 import _LANGUAGE, _EXPERIMENT_PARAMETERSPEC_TYPE


class SpearmintWebApp(Flask):
    def set_experiment_config(self, expt_config):
        self.experiment_config = expt_config
        self.experiment_dir = os.path.dirname(os.path.realpath(expt_config))

    def experiment_grid(self):
        return ExperimentGrid(self.experiment_dir)

    def experiment(self):
        return load_experiment(self.experiment_config)

    def experiment_results(self, grid):
        completed             = grid.get_complete()
        grid_data, vals, durs = grid.get_grid()
        worst_to_best         = sorted(completed, key=lambda i: vals[i], reverse=True)
        return worst_to_best, [vals[i] for i in worst_to_best]


app = SpearmintWebApp(__name__)

# Web App Routes

@app.route("/status")
def status():
    grid = None
    try:
        grid = app.experiment_grid()
        job_ids, scores = app.experiment_results(grid)
        stats  = {
            'candidates': grid.get_candidates().size,
            'pending':    grid.get_pending().size,
            'completed':  grid.get_complete().size,
            'broken':     grid.get_broken().size,
            'scores':     json.dumps(scores)
        }
        return render_template('status.html', stats=stats)
    finally:
        # Make sure we unlock the grid so as not to hold up the experiment
        if grid:
            del grid


@app.route("/")
def home():
    exp = app.experiment()
    params = []
    for p in exp.variable:
        param = {
            'name': p.name,
            'min': p.min,
            'max': p.max,
            'type': _EXPERIMENT_PARAMETERSPEC_TYPE.values_by_number[p.type].name,
            'size': p.size
        }
        params.append(param)

    ex = {
        'name': exp.name,
        'language': _LANGUAGE.values_by_number[exp.language].name,
        'params': params
    }
    return render_template('home.html', experiment=ex)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "No experiment configuration file passed as argument to web app!"
        print "Usage:\n\tpython spearmint/web/app.py path/to/config.pb\n"
        sys.exit(0)

    app.set_experiment_config(sys.argv[1])
    app.run(debug=True)


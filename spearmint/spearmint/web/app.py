import os
import sys
import json
import numpy as np
import importlib
from flask import Flask, render_template, redirect, url_for, Markup

from spearmint.ExperimentGrid import ExperimentGrid
from spearmint.helpers import load_experiment
from spearmint.spearmint_pb2 import _LANGUAGE, _EXPERIMENT_PARAMETERSPEC_TYPE

class SpearmintWebApp(Flask):
    def set_experiment_config(self, expt_config):
        self.experiment_config = expt_config
        self.experiment_dir = os.path.dirname(os.path.realpath(expt_config))

    def set_chooser(self, chooser_module, chooser):
        module  = importlib.import_module('chooser.' + chooser_module)
        self.chooser = chooser

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
        (best_val, best_job) = grid.get_best()
        best_params = list()
        for p in grid.get_params(best_job):
            best_params.append('<br />' + p.name + ':')
            if p.int_val:
                best_params[-1] += np.array_str(np.array(p.int_val))
            if p.dbl_val:
                best_params[-1] += np.array_str(np.array(p.dbl_val))
            if p.str_val:
                best_params[-1] += np.array_str(np.array(p.str_val))
        dims = len(best_params)
        best_params = Markup(','.join(best_params).encode('ascii'))

        # Pump out all experiments (parameter sets) run so far
        all_params = list()
        for job,score in zip(job_ids, scores):
            all_params.append('<tr><td>' + str(job) + '</td>' +
                              '<td>' + str(score) + '</td><td>')
            sub_params = list()
            for p in grid.get_params(job):
                sub_params.append(p.name + ':')
                if p.int_val:
                    sub_params[-1] += np.array_str(np.array(p.int_val))
                if p.dbl_val:
                    sub_params[-1] += np.array_str(np.array(p.dbl_val))
                if p.str_val:
                    sub_params[-1] += np.array_str(np.array(p.str_val))
            all_params.append(',<br />'.join(sub_params).encode('ascii') +
                              '</td></tr>')
        all_params = Markup(' '.join(all_params))

        # If the chooser has a function generate_stats_html, then this
        # will be fed into the web display (as raw html).  This is handy
        # for visualizing things pertaining to the actual underlying statistic
        # model - e.g. for sensitivity analysis.
        try:
            chooseroutput = Markup(app.chooser.generate_stats_html())
        except:
            chooseroutput = ''
        stats  = {
            'candidates'    : grid.get_candidates().size,
            'pending'       : grid.get_pending().size,
            'completed'     : grid.get_complete().size,
            'broken'        : grid.get_broken().size,
            'scores'        : json.dumps(scores),
            'best'          : best_val,
            'bestparams'    : best_params,
            'chooseroutput' : chooseroutput,
            'allresults'    : all_params,
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
        'params': params,
        }
    return render_template('home.html', experiment=ex)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "No experiment configuration file passed as argument to web app!"
        print "Usage:\n\tpython spearmint/web/app.py path/to/config.pb\n"
        sys.exit(0)

    app.set_experiment_config(sys.argv[1])
    app.run(debug=True)


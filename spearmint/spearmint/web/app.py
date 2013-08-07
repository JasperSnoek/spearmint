import os
import sys
sys.path.append(os.getcwd() + '/../lib')

import json
from ExperimentGrid import ExperimentGrid
from flask import Flask, render_template, redirect, url_for

class SpearmintWebApp(Flask):
    def set_experiment_dir(self, expt_dir):
        self.experiment_dir = expt_dir

    def get_experiment_dir(self):
        return self.experiment_dir

app = SpearmintWebApp(__name__)

def experiment_result_vals(grid):
    completed             = grid.get_complete()
    grid_data, vals, durs = grid.get_grid()
    worst_to_best         = sorted(completed, key=lambda i: vals[i], reverse=True)
    return worst_to_best, [vals[i] for i in worst_to_best]


@app.route("/status")
def status():
    try:
        expt_dir = app.get_experiment_dir()
        grid   = ExperimentGrid(expt_dir)
        xs, ys = experiment_result_vals(grid)
        xs = map(int, xs)
        ys = map(float, ys)
        stats  = {'completed':  grid.get_complete().size,
                  'pending':    grid.get_pending().size,
                  'candidates': grid.get_candidates().size,
                  'broken':     grid.get_broken().size,
                  'xs':         json.dumps(xs),
                  'ys':         json.dumps(ys)
                 }
        return render_template('status.html', **stats)
    finally:
        # Make sure we unlock the grid so as not to hold up the experiment
        del grid


@app.route("/")
def home():
    expt = load_experiment
    return render_template('home.html')
    #return redirect(url_for('static', filename='index.html'))


if __name__ == "__main__":
    import os
    app.set_experiment_dir(os.getcwd() + '/../examples/braninpy')
    app.run(debug=True)


import sys, os
sys.path.insert(0, '.')
import fig_config as _fc
_fc.OUTPUT_DIR = './figures'

import qdiff_plots_v2 as qp

rd_results = qp.load_rd('./results/realdata_results.json')
print('Loaded', len(rd_results), 'datasets')

print('Figure 3 -- KM curves...')
print(' Saved:', qp.plot_figure3(rd_results))

print('Figure 4 -- Quantile difference bands...')
print(' Saved:', qp.plot_figure4(rd_results))

print('Figures 5a/5b/5c -- EL ratio...')
for path in qp.plot_figure5(rd_results):
    print(' Saved:', path)

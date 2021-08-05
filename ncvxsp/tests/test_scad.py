# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 19:58:19 2021

@author: Clément Lejeune (clement.lejeune@irit.fr; clementlej@gmail.com)
"""

from sklearn.utils.estimator_checks import check_estimator
from ncvxlm.ncvxlm.linear_model.scad_coordinate_descent import SCADnet, MultiTaskSCADnet

## test of SCADnet
scad = SCADnet(scad_ratio=1.0)
check_estimator(scad) # passes

scad = SCADnet(scad_ratio=0.8)
check_estimator(scad) # passes

## test of MultiTaskSCADnet
### SCAD(||w_i||_1, gam)
scad_mt_gl1 = MultiTaskSCADnet(task_sparsity='group-l1')
check_estimator(scad_mt_gl1) # passes

### SCAD(||w_i||_2, gam)
scad_mt_gl2 = MultiTaskSCADnet(scad_ratio=1.0, gam=2.5, task_sparsity='group-l2')
check_estimator(scad_mt_gl2) # passes

scad_mt_gl2 = MultiTaskSCADnet(scad_ratio=0.51, gam=2.5, task_sparsity='group-l2')
check_estimator(scad_mt_gl2) # passes

scad_mt_gl2 = MultiTaskSCADnet(scad_ratio=0.50, gam=2.5, task_sparsity='group-l2')
check_estimator(scad_mt_gl2) # does not pass:
# scad_ratio <= 0.5, => zero division in prox_l2, => nn=||tmp||_2=0.0; why ?

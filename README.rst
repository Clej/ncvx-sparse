.. rst:directive:: math

ncvx-sparse
===========

**ncvx-sparse** is a Python library for learning high-dimensional linear regresion models (single- and -multi-task) with nonconvex sparsity (e.g. SCAD, MCP, l1-group SCAD).
Solvers are written in Cython and implementation follows the Scikit-learn API.

Why imposing sparsity with nonconvex sparsity inducing penalties (e.g. LASSO) ? Because...

Currently, the **ncvx-sparse** solves the following problems:

1. Single-task linear regression,

.. math::

			\arg \min_{\beta \in \mathbb{R}^p} \frac{1}{2n} \sum_i (y_i - x_i^{\top} \beta)^2 + \lambda \rho P(\beta) + \frac{1-\rho}{2} ||\beta||_2^2

where P stands for:

- SCAD (SCADnet estimator), with parameter $\gamma > 2$.

2. Multi-task linear regression,

.. math::

			\arg \min_{\beta = (\beta_1 \dots \beta_k) \in \mathbb{R}^{K \times p}} \frac{1}{2} \sum_j^K \sum_i^n (y_{ik} - x_{ik}^{\top} \beta_j)^2

where P stands for:

- SCAD-l1 i.e. SCAD on the l1-norm of p-th feature vector accross the K tasks,
- SCAD-l2, same as SCAD-l1 but with respect to the l2-norm (not squared).

Install the released version
============================

Create a Python=3.6 environment (e.g. Anaconda), and install ncvx-sparse `from pip <https://pypi.python.org/pypi/ncvx-sparse/>`__.  with the following command line in your Anaconda prompt:

::

    pip install -U ncvx-sparse
	
Example
=======

References
==========


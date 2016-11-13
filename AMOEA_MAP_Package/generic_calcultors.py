# ------------------------------------------------------------------------------
# Copyright (C) 2016 Aras Ahmadi
# Institut National des Sciences Appliquees de Toulouse (INSA Toulouse), LISBP.
# AMOEA-MAP - A Platform for the optimization of expensive multi-objective problems.
# MAP - An algorithm for the memory-based adaptive partitioning of the search space
# in Pareto-based multi-objective optimization.
#
# As regards AMOEA-MAP and MAP, the permission to use, copy, modify, and distribute
# these algorithms and their documentation for academic and research purposes, without fee,
# is hereby granted under the condition of citing the corresponding references mentioned
# below [1,2], provided that the above copyright notice and the following paragraphs appear
# in all copies of the programs.
#
# IN NO EVENT SHALL THE AUTHOR AND THE RELATED INSTITUTIONS BE LIABLE TO ANY PARTY FOR
# DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE
# USE OF THESE ALGORITHMS AND THEIR DOCUMENTATION, EVEN IF THEY HAVE BEEN ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
# THE AUTHOR AND THE RELATED INSTITUTIONS DISCLAIM ANY WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE. THE ALGORITHMS PROVIDED HEREUNDER IS ON AN "AS IS" BASIS,
# AND THE AUTHOR AND THE RELATED INSTITUTIONS HAVE NO OBLIGATION TO PROVIDE MAINTENANCE,
# SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
#
# [1] Ahmadi et al., 2016. An archive-based multi-objective evolutionary algorithm with
# adaptive search space partitioning to deal with expensive optimization problems:
# application to process eco-design, Computers and Chemical Engineering 87 (2016) 95-110.
# [2] Ahmadi A., 2016. Memory-based Adaptive Partitioning (MAP) of search space for the
# enhancement of convergence in Pareto-based multi-objective evolutionary algorithms,
# Journal of Applied Soft Computing 41 (2016) 400-417.
# ------------------------------------------------------------------------------

""" Generic calculators """

from HV_dimension_sweep import HyperVolume

# integer adjustment
def integer_adjustment(x_in):
    up_ = 1.0
    low_ = 0.0
    [a,b] = divmod(x_in,up_)
    if (up_ - b)<(b - low_):
        x_out = a + 1
    else:
        x_out = a
    return x_out

# variable bounding
def bounded(x, Up, Lo):
    if x > Up:
        x = Up
    elif x < Lo:
        x = Lo
    return x

# Hyper volume calculations by Dimension Sweep approach
def hyper_volume_Dsweep(P_,arg):
    referencePoint = arg['Reference point for HVI(DS)']
    hyperVolume = HyperVolume(referencePoint)
    hyper_volume_DS = hyperVolume.compute(P_) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return hyper_volume_DS

# IGD
def convergence(first_front, optimal_front):
    distances = []
    for opt_ind in optimal_front:
        distances.append(float("inf"))
        for ind in first_front:
            dist = 0.
            for i in xrange(len(opt_ind)):
                dist += (ind.objective_value[i] - opt_ind[i])**2
            if (dist)**0.5 < distances[-1]:
                distances[-1] = (dist)**0.5
    sum_ = 0
    for i_dist in distances:
        sum_ = sum_ + i_dist
    return (sum_) / len(optimal_front)


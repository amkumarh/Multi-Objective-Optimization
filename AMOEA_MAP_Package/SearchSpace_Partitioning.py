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

""" Adaptive Partitioning of Search Space """

from math import log
from generic_calcultors import *

# Restriction of solutions
def Solutions_Restriction(var,arg):
    for ind in range(arg["Number of variables"]):
        if arg['Ubounds'][ind]==arg['Lbounds'][ind]:
            var[ind] = arg['Lbounds'][ind]
        else:
            dx = (arg['Ubounds'][ind]-arg['Lbounds'][ind])/arg["MAP"][1][ind]
            temp1 = integer_adjustment((var[ind]-arg['Lbounds'][ind])/dx)
            temp2 = temp1/arg["MAP"][1][ind]
            temp2 = temp2*(arg['Ubounds'][ind]-arg['Lbounds'][ind])+arg['Lbounds'][ind]
            var[ind] = round(temp2,12)
    return var

# variable refining
def Dynamic_Parameter_Refining(R_,up_,arg):
    Lower_hist = []
    Higher_hist = []
    sensitivity_hist = []
    for v_ in range(arg["Number of variables"]):
        # variable normalizing
        VAR = []
        del VAR[:]
        for c_ in range(len(R_)):
            VAR.append((R_[c_].variables[v_]-R_[c_].Lsup[v_])/(R_[c_].Usup[v_]-R_[c_].Lsup[v_]))
        Lower_hist.append(min(VAR))
        Higher_hist.append(max(VAR))
        sensitivity_hist.append(max(VAR)-min(VAR))

    basis_ = max(sensitivity_hist)
    multiplier = 2
    Importance = []
    for v_ in range(arg["Number of variables"]):
        Importance_ = (Higher_hist[v_]-Lower_hist[v_])/max(sensitivity_hist)
        Importance.append(Importance_)

        l_int = log(arg['MAP'][2][0]/arg['MAP'][2][0])/log(multiplier)
        if up_ < arg['MAP'][2][1]:
            h_int = log(up_/arg['MAP'][2][0])/log(multiplier)
        else:
            h_int = log(arg['MAP'][2][1]/arg['MAP'][2][0])/log(multiplier)
        arg["MAP"][1][v_] = int(arg['MAP'][2][0]*(multiplier**(integer_adjustment(Importance_*(h_int-l_int)) + l_int)))

    return Lower_hist, Higher_hist, Importance



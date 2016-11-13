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

"""
AMOEA-MAP: main program

INPUTS (arg):
Population size, Number of variables, Number of objectives, Benchmark,
Crossover [method, probability], Mutation [method, probability]

Reference point for HVI(DS) [R1,R2,...]: reference point over the optimal
Pareto front to define

Termination criteria: Genenration Max (maximum number of genetic generations),
Max function calls (computational budget)

MAP (Memory-based Adaptive Partitioning parameters):
[Activate(True/False), [], [PTmin, PTmax]]

Expensive (True/False): expensive optimization with fixed computational budget
in terms of function calls (Automatic use of Archive-based MOEA and IAMO
for mutation)

Available Benchmarks: ZDT1_2D, DTLZ1_3D, Beam design

OUTPUT 1: HVI(dimension sweap), function calls
OUTPUT 2: Pareto front in AMOEA_MAP_Pareto_Fronts.csv
"""

import os,sys
import copy
import csv
import json

dirname, filename = os.path.split(os.path.abspath(__file__))
package_path = dirname+'\\AMOEA_MAP_Package'
sys.path.append(package_path)

from generic_calcultors import *
from SearchSpace_Partitioning import *
from AMOEA_MAP import Chromosome
from AMOEA_MAP import AMOEA_MAP_framework
from MOO_functions_ import *
from AMOEA_MAP import MultiObjectives

arg = {
        "Population size" : 30,
        "Number of variables" : 10,
        "Number of objectives" : 2,
        "MAP": [False, [], []],
        "Expensive": False,
        "Constraints": True,
        "Constraint method": "Dynamic",
        "Number of constraints": 8,
        "Genenration Max" : 3000,
        "Max function calls": 10000,
        "Crossover" : ["SBX", 1.],
        "Mutation" : ["Polynomial", None],
        "Reference point for HVI(DS)" : [10000,10000],
    }
Benchmark = G7_2D(arg["Number of variables"])
arg['Ubounds'] = Benchmark.Ubounds
arg['Lbounds'] = Benchmark.Lbounds

# Expensive MOPs
arg['Archive Pareto size'] = arg['Population size']
if arg['Expensive']:
    arg['Mutation'][1] = 0.05
    arg['Mutation'][0] = "IAMO"
    arg['Archive strategy'] = True
    arg["Population size"]= 8
else:
    arg['Mutation'][1] = 1./float(arg['Number of variables'])
    arg['Archive strategy'] = False
    arg["Population size"]= arg["Archive Pareto size"]

# common arg
arg["Main program path"] = dirname
arg["Results"] = [0,0,0,0,[],[]]
# Pmin, Pmax
arg["MAP"][2] = [10,320]

# parameter initialization
for i_ in range(arg["Number of variables"]):
    arg["MAP"][1].append(arg['MAP'][2][0])
tot_func_calls = []
LS_func_calls = []
GD_measure = []
IH_difference = []
diversities = []
Initialization_distance = []
Initialization_IHV = []
Benchmark.Bench_descret_matrix = {}


if arg['Constraint method']=='APM': arg["Average values"] =[-1e6,-1e6,0,0,0,0]      #this is the average values for the first population

                                                                                            #start is objective and last is the constraint
                                                                                            #average objective value is chosen as the best objective value
                                                                                            #and for the average constraint violation it is chosen as 0(all feasible in start)



if arg['Constraint method']=='Sranking':arg["probability factor"] = 0.450               #for the best results choose probability factor=0.450
                                                                            #use pf <0.5 so as to make search bias against infeasible solutions



if arg['Constraint method']=='Sadaptive':
    arg["best"]=[-1e6,0.0,-1e6,0.0]                          #stores the best individual based on the objective function value for each objective
                                                #and also stores the infeasibility measure if there is no feasible then based on the infeasibility measure
    arg["worst"]=[1e6,1.0,1e6,1.0]                         #stores the worst individual based on the objective function value for each objective
                                                #and also stores the infeasibility measure
    arg["highest"]=[1e6,1e6]                  #stores the maximum objective function value for each of the objective function
    arg["max_constraint"]=[1e9,1e9,1e9,1e9]
    arg["any_feasible"]=0



if arg["Constraint method"]=="alpha":       #just for the start consider alpha=0
    arg["alpha"]=0.1                           #for starting value of alpha = (max ST level of any solution + average of ST level of all solution)/2s
    arg["Satisfaction"]=[]


for i_ in range(arg["Number of variables"]):
    arg['MAP'][1][i_] = arg['MAP'][2][0]

P = []
Pareto_archive = []
arg["Results"][1] = 0
arg["Results"][0] = 0
arg["Results"][2] = 0
arg["Results"][3] = 0
arg["Results"][4] = []
arg["Results"][5] = []

# framework call
Hybrid_Optimization = AMOEA_MAP_framework(arg)

# ==============================================================================
# Population initilization
def Automatic_Initialization(arg):
    pop_size = arg["Population size"]
    n_var = arg["Number of variables"]
    for k in range(pop_size):
        P.append(Chromosome(arg))
        for m in range(n_var):
            u_ = arg['Ubounds'][m]
            l_ = arg['Lbounds'][m]
            delta_ = (u_-l_)/(pop_size-1)               #here decrease the stepsize for each of the variable according the 40% formula
            P[k].variables[m]=float(l_+k*delta_)

        # integer projection of variable space
        if arg["MAP"][0]:
            P[k].variables = Solutions_Restriction(P[k].variables,arg)

    return P

P = Automatic_Initialization(arg)

for s in P:
    s.evaluation(Benchmark,P,arg)
    if s.evaluated:
        arg["Results"][1] = arg["Results"][1] + 1
# ==============================================================================

# AMOEA-MAP framewor startup
print 'HVI \t\t Function calls'
P_out = Hybrid_Optimization.start(P, Pareto_archive, arg, Benchmark)
P_out.sort(key=lambda x: x.fitness[0])

# Save optimal Pareto front
csv_file = open(dirname+'\\AMOEA_MAP_Pareto_Fronts.csv', 'w')
for i in range(len(P_out)):
    for k in range(arg["Number of variables"]):
        csv_file.write(" " + str(P_out[i].variables[k]) + ", ")
    for j in range(arg["Number of objectives"]):
        csv_file.write(" " + str(P_out[i].fitness.values[j]) + ", ")
    csv_file.write(" " + str(P_out[i].constraint) + "\n")
csv_file.close()

print '\nObjective function was optimized through ' + str(int(arg['Max function calls'])) + \
        ' function calls \n' +  'Optimal Pareto results in AMOEA_MAP_Pareto_Fronts.csv \n' + \
        '--------------------------------------------------------------------------- \n' + \
        'AMOEA-MAP: a tool for expensive multiobjective optimization \n[1] Ahmadi A., Applied Soft Computing 41 (2016) 400-417 \n' + \
        '[2] Ahmadi et al., Computers and Chemical Engineering 87 (2016) 95-110 \n'





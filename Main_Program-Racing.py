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

import time
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
Initial_Time = time.time()

arg = {
        "Population size" : 30,
        "Number of variables" : 10,
        "Number of objectives" : 2,
        "Number of constraints": 8,
        "Genenration Max" : 3000,
        "Max function calls": 10000,
        "Constraint method": "APM",                             #list of methods available ---  1. Dynamic 2. APM 3. Self Adaptive(Sadaptive) 4. Stochastic Ranking(Sranking)  5. Constraint Dominancy(Constraint_dominancy)
        "Reference point for HVI(DS)" : [10000,10000],
        "MAP": [False, [], []],
        "Expensive": False,
        "Constraints": True,
        "Crossover" : ["SBX", 1.],
        "Mutation" : ["Polynomial", None],
    }
Benchmark = G7_2D(arg["Number of variables"])                           #   1. beam_design_2D  2. OSY_2D  3. G7_3D 4. G7_2D 5. G19_3D 6. BICOP1_2D
arg['Ubounds'] = Benchmark.Ubounds
arg['Lbounds'] = Benchmark.Lbounds
history = []
comparison_file = open(dirname+'\\comparison_file.dat', 'w')
comparison_file.close()

comparison_file2 = open(dirname+'\\comparison_file2.dat', 'w')
comparison_file2.write("case = Bench('"+ str(Benchmark.Name) +"')" + "\n")
comparison_file2.write("case.results.FunctionCalls =" + str(arg["Max function calls"])+"\n")
comparison_file2.close()

method = arg['Constraint method']
iterate = 1


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
arg["Results"] = [0,0,0,0,[],[],[]]
# Pmin, Pmax
arg["MAP"][2] = [10,320]

# parameter initialization
for i_ in range(arg["Number of variables"]):
    arg["MAP"][1].append(arg['MAP'][2][0])
tot_func_calls = []
LS_func_calls = []
GD_measure = []
IH_difference = []
Const_viol_last = []
diversities = []
Initialization_distance = []
Initialization_IHV = []
speed_IHV = []
speed_IGD = []
Fitnesses = []
speed_constraint = []

for stud_i in range(iterate):

    if arg['Constraint method']=='APM':
        arg["Average values"]=[]
        for i in range(arg["Number of objectives"]):            #here is the initialization of average values of objective function for first population(first generation)
            arg["Average values"].append(0)
        for i in range(arg["Number of constraints"]):          # #this is the initialization of constraint violation for each constraint for first population(first generation)
            arg["Average values"].append( 1e9 )
                                                            #average objective value is chosen as the best objective value
                                                            #and for the average constraint violation it is chosen as 1e9 (all infeasible in start)


    if arg['Constraint method']=='Sranking':
        arg["probability factor"] = 0.4             #for the best results choose probability factor (Pf)=0.450
                                                    #use pf <0.5 so as to make search bias against infeasible solutions


    if arg['Constraint method']=='Sadaptive':
        arg["best"]=[]
        for i in range(arg["Number of objectives"]):        #stores the best individual based on the objective function value for each objective
            arg["best"].append(0 )                          #best individual is feasible so infeasibility measure = 0
            arg["best"].append( 0.0 )
        arg["worst"]=[]
        for i in range(arg["Number of objectives"]):              #stores the worst individual based on the objective function value for each objective
            arg["worst"].append( 1e5 )                              #worst is infeasible so infeasibility measure = 1.0
            arg["worst"].append( 1.0 )
        arg["highest"]=[]
        for i in range(arg["Number of objectives"]):                #stores the maximum objective function value for each of the objective function
            arg["highest"].append(  1e6 )
        arg["max_constraint"]=[]
        for i in range(arg["Number of constraints"]):                       #stores the max constraint violation for the current generation
            arg["max_constraint"].append(  1e9 )

        arg["any_feasible"]=0


    if arg["Constraint method"]=="alpha":           #just for the start consider alpha=0
        arg["alpha"]=0.1                           #for starting value of alpha = (max ST level of any solution + average of ST level of all solution)/2
        arg["Satisfaction"]=[]

    Benchmark.Bench_descret_matrix = {}
    for i_ in range(arg["Number of variables"]):
        arg['MAP'][1][i_] = arg['MAP'][2][0]

    Hybrid_Optimization = AMOEA_MAP_framework(arg)
    memory_file = open(dirname+'\\memory.dat', 'w')
    memory_file.close()

    P = []
    Pareto_archive = []
    arg["Results"][1] = 0
    arg["Results"][0] = 0
    arg["Results"][2] = 0
    arg["Results"][3] = 0
    arg["Results"][4] = []
    arg["Results"][5] = []
    arg["Results"][6] = []

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
                delta_ = (u_-l_)/(pop_size-1)
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

    optimal_front = json.load(open(dirname+Benchmark.address))
    hyper_volume_DS_Pareto = hyper_volume_Dsweep(optimal_front,arg)

    # AMOEA-MAP framewor startup
    P_out = Hybrid_Optimization.start(P, Pareto_archive, arg, Benchmark)
    P_out.sort(key=lambda x: x.fitness[0])

    # Calculation time
    calc_time = (time.time() - Initial_Time)
    print calc_time," seconds"
    #print i

    tot_func_calls.append(arg["Results"][1])
    GD_measure.append(arg["Results"][2])
    IH_difference.append(hyper_volume_DS_Pareto-arg["Results"][3])
    len_temp = len(arg["Results"][6])
    Const_viol_last.append(arg["Results"][6][len_temp-1][0])
    for k_ in arg["Results"][4]:
        k_[0] = hyper_volume_DS_Pareto-k_[0]
    speed_IHV.append(arg["Results"][4])
    speed_IGD.append(arg["Results"][5])
    speed_constraint.append(arg["Results"][6])

    temp = []
    for sol_ in P_out:
        temp.append(sol_.objective_value)
    Fitnesses.append(temp)

comparison_file2 = open(dirname+'\\comparison_file2.dat', 'a')
comparison_file2.write("## P E R F O R M A N C E" + "\n")
temp_ = []
for ii_ in range(len(GD_measure)):
    temp_.append(abs(GD_measure[ii_]-sum(GD_measure)/iterate))
best_fitness_IGD = Fitnesses[temp_.index(min(temp_))]
best_speed_IGD = speed_IGD[temp_.index(min(temp_))]
best_speed_Constraint_IGD = speed_constraint[temp_.index(min(temp_))]
temp_ = []
for ii_ in range(len(IH_difference)):
    temp_.append(abs(IH_difference[ii_]-sum(IH_difference)/iterate))
best_fitness_IHV = Fitnesses[temp_.index(min(temp_))]
best_speed_IHV = speed_IHV[temp_.index(min(temp_))]
best_speed_Constraint_IHV = speed_constraint[temp_.index(min(temp_))]


# write to file
comparison_file2.write("case.results.IGD." + str(method) + ".append(" + str(GD_measure)+")"+"\n")
comparison_file2.write("case.results.IHV." + str(method) + ".append(" + str(IH_difference)+")"+"\n")
comparison_file2.write("## P A R E T O   F R O N T (IGD)" + "\n")
comparison_file2.write("case.results.ParetoIGD." + str(method) + " = [")
for ii_ in range(len(best_fitness_IGD)):
    if ii_<>(len(best_fitness_IGD)-1):
        comparison_file2.write(str(best_fitness_IGD[ii_]) +",")
    else:
        comparison_file2.write("]"+ "\n")
comparison_file2.write("## P A R E T O   F R O N T (IHV)" + "\n")
comparison_file2.write("case.results.ParetoIHV." + str(method) + " = [")
for ii_ in range(len(best_fitness_IHV)):
    if ii_<>(len(best_fitness_IHV)-1):
        comparison_file2.write(str(best_fitness_IHV[ii_]) +",")
    else:
        comparison_file2.write("]"+ "\n")
comparison_file2.write("## S P E E D (IGD) " + "\n")
comparison_file2.write("case.results.SpeedIGD." + str(method) + " = [")
for ii_ in range(len(best_speed_IGD)):
    if ii_<>(len(best_speed_IGD)-1):
        comparison_file2.write(str(best_speed_IGD[ii_]) +",")
    else:
        comparison_file2.write("]"+ "\n")
comparison_file2.write("## S P E E D (IHV) " + "\n")
comparison_file2.write("case.results.SpeedIHV." + str(method) + " = [")
for ii_ in range(len(best_speed_IHV)):
    if ii_<>(len(best_speed_IHV)-1):
        comparison_file2.write(str(best_speed_IHV[ii_]) +",")
    else:
        comparison_file2.write("]"+ "\n")
comparison_file2.write("## S P E E D (IGD) (Constraint violation) " + "\n")
comparison_file2.write("case.results.SpeedConstraint." + str(method) + " = [")
for ii_ in range(len(best_speed_Constraint_IGD)):
    if ii_<>(len(best_speed_Constraint_IGD)-1):
        comparison_file2.write(str(best_speed_Constraint_IGD[ii_]) +",")
    else:
        comparison_file2.write("]"+ "\n")
comparison_file2.write("## S P E E D (IHV) (Constraint violation) " + "\n")
comparison_file2.write("case.results.SpeedConstraint." + str(method) + " = [")
for ii_ in range(len(best_speed_Constraint_IHV)):
    if ii_<>(len(best_speed_Constraint_IHV)-1):
        comparison_file2.write(str(best_speed_Constraint_IHV[ii_]) +",")
    else:
        comparison_file2.write("]"+ "\n")

comparison_file2.close()

comparison_file = open(dirname+'\\comparison_file.dat', 'a')
comparison_file.write("# Method: "+method + "\n")
comparison_file.write("  start, "+"max, "+"min, "+"end, "+"average "+ "\n")
comparison_file.write("# Function calls: "+ str(tot_func_calls[0])+", "+str(max(tot_func_calls))+", "+str(min(tot_func_calls))+", "+str(tot_func_calls[iterate-1])+", "+str(sum(tot_func_calls)/iterate)+"\n")
comparison_file.write("# GD: "+ str(GD_measure[0])+", "+str(max(GD_measure))+", "+str(min(GD_measure))+", "+str(GD_measure[iterate-1])+", "+str(sum(GD_measure)/iterate)+"\n")
comparison_file.write("# IH (Dimension Sweep): "+ str(IH_difference[0])+", "+str(max(IH_difference))+", "+str(min(IH_difference))+", "+str(IH_difference[iterate-1])+", "+str(sum(IH_difference)/iterate)+"\n")
comparison_file.write("# Mean constraint violations: "+ str(Const_viol_last[0])+", "+str(max(Const_viol_last))+", "+str(min(Const_viol_last))+", "+str(Const_viol_last[iterate-1])+", "+str(sum(Const_viol_last)/iterate)+"\n")
comparison_file.write("convergence_GD: "+ str(sum(GD_measure)/iterate)+ "\n")
comparison_file.write("\n")
comparison_file.write("# Function calls: "+ str(tot_func_calls)+"\n")
comparison_file.write("# GD: "+ str(GD_measure)+"\n")
comparison_file.write("# IH (Dimension Sweep): "+ str(IH_difference)+"\n")
comparison_file.write("# Mean constraint violations: "+ str(Const_viol_last)+"\n")
temp_ = []
for ii_ in range(len(GD_measure)):
    temp_.append(abs(GD_measure[ii_]-sum(GD_measure)/iterate))
best_fitness = Fitnesses[temp_.index(min(temp_))]
comparison_file.write("# Best fitness:" + "\n")
for ii_ in range(len(best_fitness)):
    comparison_file.write(" "+ str(best_fitness[ii_]) +"\n")
comparison_file.write("====================="+"\n")
comparison_file.close()



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





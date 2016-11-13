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

""" AMOEA-MAP """

import sys, random, math, copy
import json
from MOO_functions_ import *
from operators import *
from generic_calcultors import *
from SearchSpace_Partitioning import *

class Chromosome():
    def __init__(self, arg):
        self.arg = arg
        self.evaluated = None
        self.Num_Obj = arg["Number of objectives"]
        self.variables = []
        self.maximize = False
        self.Lsup = arg["Lbounds"]
        self.Usup = arg["Ubounds"]
        self.rank = sys.maxint
        self.distance = 0.0
        self.Num_Var = arg["Number of variables"]
        for i in range(self.Num_Var):
            self.variables.append(random.random()*(self.Usup[i]-self.Lsup[i]) + self.Lsup[i])
        # MAP
        if arg["MAP"][0]:
            self.variables = Solutions_Restriction(self.variables,arg)
        self.Crossover_method = arg["Crossover"][0]
        self.Mutation_method = arg["Mutation"][0]
        self.Cross_rate = float(arg["Crossover"][1])
        self.Mute_rate = float(arg["Mutation"][1])
        objective_vector = []
        del objective_vector[:]
        for k in range(self.Num_Obj):
            objective_vector.append(0.0)
        self.fitness = MultiObjectives(objective_vector,0.0,arg)
        self.constraint = 0.0
        self.constraint_value=[]                                        #this list will store the constrain violation for the individual
        self.objective_value=[]                                         #this list will store the objective function value for the individual

    def __setattr__(self, name, val):
        if name == 'variables':
            self.__dict__[name] = val
            #self.fitness = None
        else:
            self.__dict__[name] = val
    def __str__(self):
        return '%s : %s' % (str(self.variables), str(self.fitness), str(self.constraint))
    def __repr__(self):
        return '<Individual: candidate = %s, fitness = %s, constraint = %s>' % ( str(self.variables), str(self.fitness), str(self.constraint))
    def __lt__(self, other):
        if self.fitness is not None and other.fitness is not None:
            if self.maximize:
                return self.fitness < other.fitness
            else:
                return self.fitness > other.fitness
        else:
            raise Exception('fitness is not defined')
    def __le__(self, other):
        return self < other or not other < self
    def __gt__(self, other):
        if self.fitness is not None and other.fitness is not None:
            return other < self
        else:
            raise Exception('fitness is not defined')
    def __ge__(self, other):
        return other < self or not self < other
    def __lshift__(self, other):
        return self < other
    def __rshift__(self, other):
        return other < self
    def __ilshift__(self, other):
        raise TypeError("unsupported operand type(s) for <<=: 'Individual' and 'Individual'")
    def __irshift__(self, other):
        raise TypeError("unsupported operand type(s) for >>=: 'Individual' and 'Individual'")
    def __eq__(self, other):
        return self.variables == other.variables
    def __ne__(self, other):
        return self.variables != other.variables

    def evaluation(self, Benchmark,P,arg):
        string_ind = str(self.variables)
        if string_ind in Benchmark.Bench_descret_matrix.keys() and self.arg["MAP"][0]:
            # memory use
            objective_vector = Benchmark.Bench_descret_matrix[string_ind].fitness.values
            self.constraint = Benchmark.Bench_descret_matrix[string_ind].constraint
            self.fitness = MultiObjectives(objective_vector,self.constraint, arg)
            self.constraint_value=Benchmark.Bench_descret_matrix[string_ind].constraint_value
            self.objective_value=Benchmark.Bench_descret_matrix[string_ind].objective_value
            self.evaluated = False
        else:
            # memory creation
            objective_vector = Benchmark.evaluate(self,arg)
            self.fitness = MultiObjectives(objective_vector,self.constraint, arg)
            Benchmark.Bench_descret_matrix[string_ind] = self
            self.evaluated = True

class MultiObjectives(object):
    def __init__(self, values=[], const=[], arg={}, maximize=True):
        self.values = values
        try:
            iter(maximize)
        except TypeError:
            maximize = [maximize for v in values]
        self.maximize = maximize
        self.const = const
        self.const_method = arg['Constraint method']
    def __len__(self):
        return len(self.values)
    def __getitem__(self, key):
        return self.values[key]
    def __iter__(self):
        return iter(self.values)
    def __lt__(self, other):
        if len(self.values) != len(other.values):
            raise NotImplementedError
        else:
            not_worse = True
            strictly_better = False
            domination = False

            if self.const_method in ['Constraint_dominancy']:                       #checking the dominancy for the constraint dominancy method
                x_c = self.const
                y_c = other.const
                if x_c<>0 and y_c<>0:
                    if x_c < y_c:
                        domination = True
                elif x_c==0 and y_c<>0:
                    domination = True
                elif (x_c==0 and y_c==0):
                    for x, y, m in zip(self.values, other.values, self.maximize):
                        if m:
                            if x > y:
                                not_worse = False
                            elif y > x:
                                strictly_better = True
                        else:

                            if x < y:
                                not_worse = False
                            elif y < x:
                                strictly_better = True
                    domination = not_worse and strictly_better

            else:                                                                               #checking the dominancy for other method
                for x, y, m in zip(self.values, other.values, self.maximize):
                    if m:
                        if x > y:
                            not_worse = False
                        elif y > x:
                            strictly_better = True
                    else:

                        if x < y:
                            not_worse = False
                        elif y < x:
                            strictly_better = True
                domination = not_worse and strictly_better

            return domination

    def __le__(self, other):
        return self < other or not other < self
    def __gt__(self, other):
        return other < self
    def __ge__(self, other):
        return other < self or not self < other
    def __eq__(self, other):
        return self.values == other.values
    def __str__(self):
        return str(self.values)
    def __repr__(self):
        return str(self.values)

class AMOEA_MAP_framework:
    def __init__(self, arg):
        self.Num_Obj = arg["Number of objectives"]
        self.memory_use = 0
        self.Mute_rate = float(arg["Mutation"][1])
        self.Cross_rate = float(arg["Crossover"][1])
        self.Pop_Size = arg["Population size"]
        self.Gen_Max = arg["Genenration Max"]
        self.Num_Var = arg["Number of variables"]
        self.Ref_point = arg["Reference point for HVI(DS)"]
        random.seed();

    def start(self, P, Pareto_archive, arg, Benchmark):
        Q = []
        fmax = []
        fmin = []
        temp = []
        HVI_DS = []
        mean1 = []
        const_violation = []
        Pareto_archive_survived = []
        rLower_b = []
        rHigher_b = []

        up_ = arg['MAP'][2][0]
        Importance = []
        for i_ in range(arg['Number of variables']):
            Importance.append(1.0)
        arg['Amplitude'] = []
        arg['memory use']=0

        ## Generations
        for i in range(self.Gen_Max):
            arg["Current generation"] = i

            # constraint operations
            if arg["Constraints"]:
                if arg["Constraint method"] == "APM":
                    #arg["Average values"] = []
                    arg["Average values"] = self.average(arg,P)                     #this function returns the average objective fucntion value and average constraint violation

                if arg["Constraint method"]=="Sadaptive":
                    arg["max_constraint"]=self.maximum(arg,P)                       #this function returns maximum constraint violation in the population
                    arg["best"],arg["worst"],arg["highest"]=self.adaptive(arg,P)    #this function returns three list which stores the best individual and its infeasibility measure and same for the worst and highest objective value individual


                if arg["Constraint method"]=="alpha":
                    #calculate the satisfaction level for each individual
                    if i==0:
                        a=arg["alpha"]
                    elif i< self.Gen_Max/2:
                        a=1-(1-arg["alpha"])*((1-2*i/self.Gen_Max)**2)
                    else:
                        a=1
                    arg["Satisfaction"] = self.satisfaction_level(arg,P)

            R = []
            del R[:]
            R.extend(P)
            R.extend(Q)

            # bi-population strategy in AMOEA
            del P[:]
            Pareto_archive.extend(R)
            Pareto_archive =self.fast_non_dominated_sorting(Pareto_archive,arg['Archive Pareto size'],arg)
            if arg['Archive Pareto size'] == self.Pop_Size:
                P.extend(Pareto_archive)
            else:
                Pareto_archive_survived = []
                Pareto_archive_survived = self.fast_non_dominated_sorting(Pareto_archive,self.Pop_Size,arg)
                P.extend(Pareto_archive_survived)



            ## Pareto evolution ------------------------------------------------
            stall = 3
            if self.Num_Obj>1:
                # Hyper volume calculations via dimension-sweep algorithm
                temp_P = []
                del temp_P[:]
                if arg['Archive Pareto size'] == self.Pop_Size:
                    for x_ in P:
                        temp_P.append(x_.fitness.values)
                    hyper_volume_ind1 = hyper_volume_Dsweep(temp_P,arg)
                else:
                    for x_ in Pareto_archive:
                        temp_P.append(x_.fitness.values)
                    hyper_volume_ind1 = hyper_volume_Dsweep(temp_P,arg)
                HVI_DS.append(hyper_volume_ind1)
                mean_value = sum(HVI_DS[i-stall:])/(stall+1)
            else:
                temp_mean = 0
                for x_ in P:
                    temp_mean = temp_mean + (x_.fitness.values[0]/(len(P)))
                mean1.append(temp_mean)
                mean_value = sum(mean1[i-stall:])/(stall+1)

            if arg["Constraints"]:
                temp__ = []
                for jh_ in Pareto_archive:
                    temp__.append(jh_.constraint)
                mean_const_violation = sum(temp__)/float(len(Pareto_archive))
                const_violation.append(mean_const_violation)

            # convergence calculation
            optimal_front = json.load(open(arg["Main program path"]+Benchmark.address))
            optimal_front = sorted(optimal_front[i] for i in range(0, len(optimal_front), 2))
            if arg['Archive Pareto size'] == self.Pop_Size:
                P.sort(key=lambda x: x.fitness[0])
                convergence_GD_global = convergence(P, optimal_front)
            else:
                Pareto_archive.sort(key=lambda x: x.fitness[0])
                convergence_GD_global = convergence(Pareto_archive, optimal_front)

            # normalized standard deviation
            deviation = 0.0
            if mean_value>1.e-15 and i>=stall-1:
                if self.Num_Obj>1:
                    for nu_ in range(stall+1):
                        x= mean_value-HVI_DS[i-stall+nu_]
                        if x>1e154:
                            x=1e154
                        if x<-1e154:
                            x=-1e154
                        if mean_value >1e154:
                            mean_value = 1e154
                        deviation = deviation + (x)**2.0 / mean_value**2.0
                    deviation = (deviation/stall)**0.5
                else:
                    for nu_ in range(stall+1):
                        deviation = deviation + (mean_value-mean1[i-stall+nu_])**2.0 / mean_value**2.0
                    deviation = (deviation/stall)**0.5
            else:
                deviation = 1e2
            ## -----------------------------------------------------------------

            ## Dynamic adaptive partitioning -----------------------------------
            deviation_limit = 1.e-2
            if arg["Expensive"]:
                if arg["MAP"][0]:
                    multiplier = 2
                    max_ = log(arg['MAP'][2][1]/arg['MAP'][2][0])/log(multiplier)
                    beta_ = random.random()
                    if deviation<deviation_limit:
                        k_ = log(up_/arg['MAP'][2][0])/log(multiplier)
                        if beta_ < (0.5-0.1*k_):
                            if up_<arg['MAP'][2][1]:
                                up_ = up_*multiplier
                        else:
                            if up_>arg['MAP'][2][0]:
                                up_ = up_/multiplier
                    rLower_b, rHigher_b, Importance = Dynamic_Parameter_Refining(P,up_,arg)
            else:
                if arg["MAP"][0]:
                    multiplier = 2
                    max_ = log(arg['MAP'][2][1]/arg['MAP'][2][0])/log(multiplier)+1
                    beta_ = random.random()
                    if deviation<deviation_limit:
                        if beta_>((log(up_/arg['MAP'][2][0])/log(multiplier))/max_):
                            if up_<arg['MAP'][2][1]:
                                up_ = up_*multiplier
                        else:
                            if up_>arg['MAP'][2][0]:
                                up_ = up_/multiplier
                    rLower_b, rHigher_b, Importance = Dynamic_Parameter_Refining(P,up_,arg)

            arg['Importance'] = Importance
            ## -----------------------------------------------------------------

            if arg["Results"][1]>arg['Max function calls']:
                if self.Num_Obj>1:
                    print '%.4f' % hyper_volume_ind1,' \t ',arg["Results"][1], ' \t ', convergence_GD_global, ' \t ', mean_const_violation
                    arg["Results"][4].append([hyper_volume_ind1,arg["Results"][1]])
                    arg["Results"][5].append([convergence_GD_global,arg["Results"][1]])
                    arg["Results"][6].append([mean_const_violation,arg["Results"][1]])
                    break
                else:
                    print '%.4f' % temp_mean, ' \t ', arg["Results"][1], ' \t ', mean_const_violation
                    arg["Results"][5].append([mean_value,arg["Results"][1]])
                    arg["Results"][6].append([mean_const_violation,arg["Results"][1]])
                    break

            del fmax[:]
            del fmin[:]
            for j in range(self.Num_Obj):
                del temp[:]
                for x_ in P:
                    temp.append(x_.fitness.values[j])
                fmax.append(max(temp))
                fmin.append(min(temp))

            arg["Current function calls"] = arg["Results"][1]
            if self.Num_Obj>1:
                print '%.4f' % hyper_volume_ind1,' \t ',arg["Results"][1], ' \t ', convergence_GD_global, ' \t ', mean_const_violation
                arg["Results"][4].append([hyper_volume_ind1,arg["Results"][1]])
                arg["Results"][5].append([convergence_GD_global,arg["Results"][1]])
                arg["Results"][6].append([mean_const_violation,arg["Results"][1]])
            else:
                print '%.4f' % temp_mean, ' \t ', arg["Results"][1], ' \t ', mean_const_violation
                arg["Results"][5].append([mean_value,arg["Results"][1]])
                arg["Results"][6].append([mean_const_violation,arg["Results"][1]])

            # Regenerating new children by means of genetic operators
            Q = self.regeneration(P, arg, Benchmark)

        if self.Num_Obj>1:
            print '%.4f' % hyper_volume_ind1,' \t ',arg["Results"][1]
            arg["Results"][3] = hyper_volume_ind1
        else:
            print '%.4f' % temp_mean, arg["Results"][1]

        if self.Num_Obj>1:
            arg["Results"][2] = convergence_GD_global
        else:
            arg["Results"][2] = mean_value

        if arg['Archive Pareto size'] == self.Pop_Size:
            return P
        else:
            return Pareto_archive


    def satisfaction_level(self,arg,P):                                                 #used for calculating the satisfaction level of the individuals in the alpha constraint method.
        #calculate the satisfaction level for each individual
        satis=[]
        bi=10000
        for i in range(arg["Population size"]):
            value=[]
            for j in range(arg["Number of constraints"]):

                if P[i].constraint_value[j]<=0:
                    value.append(1)
                elif P[i].constraint_value[j] <=bi:
                    value.append(1-P[i].constraint_value[j]/bi)
                else:
                    value.append(0)
            satis.append(min(value))
        return satis

                                                                                                            #validated
                                                                                                 #calculate the avarage values of the fitness value and constraint violation of each generation
    def average(self,arg,P):                                                                        #calculate average objective function value and average constraint violation and return the list
            avgf = []
            avgc = []
            for y in range(arg["Number of objectives"]):
                somme = 0
                for y_ in range(arg["Population size"]):
                    somme +=P[y_].fitness.values[y]
                avgf.append(somme / arg["Population size"])                             #avarage values of the fitness value

            for y in range(arg['Number of constraints']):                 #no of constraints
                somme = 0
                for y_ in range(arg["Population size"]):
                    somme +=P[y_].constraint_value[y]
                avgc.append(somme / arg["Population size"])                             #average constraint violation

            output = []
            for i in range(len(avgf)):
                output.append(avgf[i])
            for i in range(len(avgc)):
                output.append(avgc[i])

            return output


                                                                                                                #returns the best,worst and highest objective value individual in the current running generation
    def adaptive(self,arg,P):                                                                           #validated
        best=[]                                                                                                  #starting values store the best individual function values
        worst=[]
        highest=[]

        max_c=[]
        max_c=self.maximum(arg,P)                                                               #return max constraint violation in the population


        #calculation of best individual
        feas=[]
        for i in range(arg["Population size"]):
            if P[i].constraint==0:
                feas.append(i)                                                              #this list will store the indices of the feasible individuals

        arg["any_feasible"]=0
        if len(feas):                                                                           #calculation of the best individual in case of any feasible solution exist in the population
            arg["any_feasible"]=1                                                               #in this case the best individual will be the one with lowest fitness value
            for i in range(arg["Number of objectives"]):
                besti=feas[0]
                mini=P[feas[0]].fitness.values[i]
                for i_ in feas[1:]:
                    if P[i_].fitness.values[i]< mini and P[i_].constraint == 0:
                        mini=P[i_].fitness.values[i]
                        besti=i_

                #const=self.penalty_factor(arg,P,besti)
                best.append(mini)
                best.append(0.0)         #best individual contains best value and infeasibilty(0 because it is feasible) and same for other obj.

        else:                                                                                       #calculation of the best individual in case of no feasible solution exist in the population
            besti=0                                                                                 #in this case the best individual will be the one with lowest infeasibility measure
            const=self.penalty_factor(arg,P,0)
            for i in range(1,arg["Population size"]):
                if self.penalty_factor(arg,P,i) < const:
                    const=self.penalty_factor(arg,P,i)
                    besti=i
            for i in range(arg["Number of objectives"]):
                best.append(P[besti].fitness.values[i])
                best.append(const)                          #in best[], fitness value and penalty factor is stored for each of he objective



                                                            #calculation for highest objective function value individual regardless of the feasibility measure

        index_highest_fitness=[]
        for i in range(arg["Number of objectives"]):
            maxi=P[0].fitness.values[i]
            index_highest_fitness.append(0)
            for i_ in range(1,arg["Population size"]):
                if P[i_].fitness.values[i] > maxi:
                    maxi=P[i_].fitness.values[i]
                    index_highest_fitness[i]=i_
            highest.append(maxi)



                                                                                            #calculation for worst individual
        if len(feas)==arg["Population size"]:
            for i in range(arg["Number of objectives"]):                                    #if all the individuals are feasible then the worst will be the one with highest objective value
                worst.append(highest[i])
                worst.append(0.0)
        else:
            for i in range(arg["Number of objectives"]):                                    #this is the case where all the infeasible individuals have fitness value larger than the fitness value of the best individual
                if_all_greater=[]                                                           #the worst individual will be one with highest infeasibility measure and objective value larger than the fitness value of the best individual
                for i_ in range(arg["Population size"]):
                    if P[i_].constraint!=0 and P[i_].fitness.values[i] >= best[i*2]:
                        if_all_greater.append(i_)
                if len(if_all_greater) == arg["Population size"]-len(feas):
                    maxi=self.penalty_factor(arg,P,if_all_greater[0])
                    index=if_all_greater[0]
                    for i_ in if_all_greater:
                        if maxi < self.penalty_factor(arg,P,i_):
                            maxi= self.penalty_factor(arg,P,i_)
                            index=i_
                    worst.append(P[index].fitness.values[i])
                    worst.append(maxi)

                if len(if_all_greater) != arg["Population size"]-len(feas):
                    if_any_less=[]                                                      #this is the case where any infeasible individuals has fitness value smaller than the fitness value of the best individual
                    for i_ in range(arg["Population size"]):                            #the worst individual will be one with highest infeasibility measure and objective value smaller than the fitness value of the best individual
                        if P[i_].constraint!=0 and P[i_].fitness.values[i] <= best[i*2]:
                            if_any_less.append(i_)
                    if len(if_any_less)!=0:
                        maxi=self.penalty_factor(arg,P,if_any_less[0])
                        index=if_any_less[0]
                        for i_ in if_any_less:
                            if maxi < self.penalty_factor(arg,P,i_) :
                                maxi= self.penalty_factor(arg,P,i_)
                                index=i_
                        worst.append(P[index].fitness.values[i])
                        worst.append(maxi)
                    else:
                        worst.append(P[index_highest_fitness[i]].fitness.values[i])
                        worst.append(self.penalty_factor(arg,P,index_highest_fitness[i]))

        return best,worst,highest


                                                                             #returns the maximum constraint violation for each constraint in the current running generation
    def maximum(self,arg,P):
        max_c=[]
        for i in range(arg["Number of constraints"]):
            maxi=0
            for i_ in range(arg["Population size"]):
                if maxi<P[i_].constraint_value[i]:
                    maxi=P[i_].constraint_value[i]
            max_c.append(maxi)
        return max_c


                                                                                            #calculate the infeasibility measure for each individual
    def penalty_factor(self,arg,P,i):
        max_c=[]
        const=0
        max_c=self.maximum(arg,P)
        for i_ in range(arg["Number of constraints"]):
            if max_c[i_]!=0:
                max_c[i_]=(P[i].constraint_value[i_])/(max_c[i_])
        const=sum(max_c)/arg["Number of constraints"]
        return const



    def regeneration(self, P, arg, Benchmark):
        # Regenerate a new population
        Q = []
        # selection
        selected_parents = tournament_selection(P, self.Pop_Size, arg)
        selected_parents_var = [copy.deepcopy(i.variables) for i in selected_parents]
        # crossover
        children_crossedover_var = crossover_operator(selected_parents_var, arg)
        # mutation
        children_mutated_var = mutation_operator(children_crossedover_var, arg)
        for i in range(len(children_mutated_var)):
            child = Chromosome(arg)
            child.variables = children_mutated_var[i]
            child.evaluation(Benchmark,P,arg)
            if child.evaluated:
                arg["Results"][1] = arg["Results"][1] + 1
            else:
                arg["memory use"] = arg["memory use"] + 1
            Q.append(child)
        return Q

    def fast_non_dominated_sorting(self, combined, archive_size,arg):
        # Fast non-dominated sorting
        survivors = []
        if arg["Constraint method"] != "Sranking":
            fronts = []
            pop = set(range(len(combined)))
            while len(pop) > 0:
                front = []
                for p in pop:
                    dominated = False
                    for q in pop:
                        if combined[p] < combined[q]:
                            dominated = True
                            break
                    if not dominated:
                        front.append(p)
                fronts.append([dict(individual=combined[f], index=f) for f in front])
                pop = pop - set(front)

            for i, front in enumerate(fronts):
                if len(survivors) + len(front) > archive_size:
                    # Determine the crowding distance.
                    distance = [0 for _ in range(len(combined))]
                    individuals = front[:]
                    num_individuals = len(individuals)
                    num_objectives = len(individuals[0]['individual'].fitness)
                    for obj in range(num_objectives):
                        individuals.sort(key=lambda x: x['individual'].fitness[obj])
                        distance[individuals[0]['index']] = float('inf')
                        distance[individuals[-1]['index']] = float('inf')
                        for i in range(1, num_individuals-1):
                            distance[individuals[i]['index']] = (distance[individuals[i]['index']] +
                                                                 (individuals[i+1]['individual'].fitness[obj] -
                                                                  individuals[i-1]['individual'].fitness[obj]))

                    crowd = [dict(dist=distance[f['index']], index=f['index']) for f in front]
                    crowd.sort(key=lambda x: x['dist'], reverse=True)
                    last_rank = [combined[c['index']] for c in crowd]
                    r = 0
                    num_added = 0
                    num_left_to_add = archive_size - len(survivors)
                    while r < len(last_rank) and num_added < num_left_to_add:
                        if last_rank[r] not in survivors:
                            survivors.append(last_rank[r])
                            num_added += 1
                        r += 1
                    if len(survivors) == archive_size:
                        break
                else:
                    for f in front:
                        if f['individual'] not in survivors:
                            survivors.append(f['individual'])
        else:                                                                                           #if the method is stochastic ranking then instead of using fast non dominated sorting ,Stochstic ranking will be used.
            num_selected = archive_size
            tourn_size = 2
            pop__ = list(combined)
            selected = []
            for p in range(arg["Population size"]):
                count=0
                # no violation or no contraint use
                #selected.append(max(tourn))
                frand=random.randint(0,1000)/1000.0
                tourn=[]
                for i in range(arg["Population size"]-1):
                    tourn=[combined[i],combined[i+1]]
                    if( frand < arg["probability factor"] or (tourn[0].constraint == 0 and tourn[1].constraint == 0) ):
                        #temp=[]
                        #temp.append(max(tourn))
                        #selected.append(max(tourn))
                        if max(tourn) != tourn[0]:                                                                                  #swap the best individual to the start of the population stochastically
                            combined[i]=copy.deepcopy(tourn[1])
                            combined[i+1]=copy.deepcopy(tourn[0])
                            count+=1

                    else:
                        if tourn[0].constraint<>0 and tourn[1].constraint==0:
                            combined[i]=copy.deepcopy(tourn[1])
                            combined[i+1]=copy.deepcopy(tourn[0])
                            #selected.append(tourn[1])
                            count+=1

                        if tourn[0].constraint<>0 and tourn[1].constraint<>0:
                            if tourn[0].constraint >= tourn[1].constraint:
                                combined[i]=copy.deepcopy(tourn[1])
                                combined[i+1]=copy.deepcopy(tourn[0])
                                #selected.append(tourn[1])
                                count+=1
                if count==0:
                    break


            for i in range(arg["Population size"]):
                survivors.append(combined[i])
        return survivors

# Tournament selection with constraints
def tournament_selection(pop_, pop_size, arg):
    num_selected = pop_size
    tourn_size = 2
    pop__ = list(pop_)
    selected = []
    for p in range(num_selected):
        count=0

        if arg['Constraint method'] in  ['Constraint_dominancy','Dynamic']:
            tourn = random.sample(pop__, tourn_size)
            if tourn[0].constraint==0 and tourn[1].constraint==0:
                selected.append(max(tourn))
            if tourn[0].constraint==0 and tourn[1].constraint<>0:
                selected.append(tourn[0])
            if tourn[0].constraint<>0 and tourn[1].constraint==0:
                selected.append(tourn[1])
            if tourn[0].constraint<>0 and tourn[1].constraint<>0:
                if tourn[0].constraint <= tourn[1].constraint:
                    selected.append(tourn[0])
                else:
                    selected.append(tourn[1])


        #selection for apm and sadaptive
        elif arg['Constraint method'] in ['APM','Sadaptive']:
            tourn = random.sample(pop__, tourn_size)
            selected.append(max(tourn))


                                                                                        #define the selection method based on the satisfaction level and  objective values
        elif arg["Constraint method"]=="alpha":
            index=[]
            tourn=[]
            index = random.sample(range(num_selected), tourn_size)
            tourn.append(pop__[index[0]])
            tourn.append(pop__[index[1]])
            if tourn[0].arg["Satisfaction"][index[0]]>=arg["alpha"] and tourn[1].arg["Satisfaction"][index[1]]>=arg["alpha"]:

                if tourn[0].fitness[0] < tourn[1].fitness[0]:
                    selected.append(tourn[0])
                else:
                    selected.append(tourn[1])

            elif tourn[0].arg["Satisfaction"][index[0]]==tourn[1].arg["Satisfaction"][index[1]]:
                #select based on the best f value
                if tourn[0].fitness[0] < tourn[1].fitness[0]:
                    selected.append(tourn[0])
                else:
                    selected.append(tourn[1])

            else:
                #select best based on the u value
                if tourn[0].arg["Satisfaction"][index[0]] > tourn[1].arg["Satisfaction"][index[1]]:
                    selected.append(tourn[0])
                else:
                    selected.append(tourn[1])


        elif arg["Constraint method"]=='Sranking':                                          #select this proportion of the population and fill the remaining space with these solutions
            truncation_rate = 1.0/7.0

            for i in range(int(pop_size * truncation_rate)):
                for j in range(int(1/ truncation_rate)):
                    selected.append(pop_[i])
            remaining_ele=pop_size % (int(1/truncation_rate))
            already_inserted=0
            for i in range(remaining_ele):
                selected.insert(i+already_inserted,pop_[i])
                already_inserted  +=  int(pop_size*truncation_rate)

    return selected

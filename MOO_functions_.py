""" Some test functions """

import math,copy
from random import uniform


class G19_3D:
    def __init__(self, Nvar):
        Lbounds = []
        Ubounds = []
        del Lbounds[:]
        del Ubounds[:]
        for i in range(Nvar):
            Lbounds.append(float(0.0))
            Ubounds.append(float(10.0))
        self.Lbounds = Lbounds
        self.Ubounds = Ubounds
        self.Name = "G19-3D"

        self.Bench_descret_matrix = {}
        self.address = "\\AMOEA_MAP_Package\\pareto_fronts_json\\G19_3D.json"
        self.function_evaluation = 0                                                            #this is an extra attribute added to implement dynamic penalty method


    def evaluate(self, chromosome, arg):
        objective_vector = []
        n_var = chromosome.Num_Var
        del objective_vector[:]

        X_ = []
        del X_[:]
        for i in range(n_var):
            X_.append(chromosome.variables[i])

        e = [-15.0, -27.0, -36.0, -18.0, -12.0]
        c = [[30.0, -20.0, -10.0, 32.0, -10.0],
             [-20.0, 39.0, -6.0, -31.0, 32.0],
             [-10.0, -6.0, 10.0, -6.0, -10.0],
             [32.0, -31.0, -6.0, 39.0, -20.0],
             [-10.0, 32.0, -10.0, -20.0, 30.0]]
        d = [4.0, 8.0, 10.0, 6.0, 2.0]
        a = [[-16.0, 2.0, 0.0, 1.0, 0.0],
             [0.0, -2.0, 0.0, 0.4, 2.0],
             [-3.5, 0.0, 2.0, 0.0, 0.0],
             [0.0, -2.0, 0.0, -4.0, -1.0],
             [0.0, -9.0, -2.0, 1.0, -2.8],
             [2.0, 0.0, -4.0, 0.0, 0.0],
             [-1.0, -1.0, -1.0, -1.0, -1.0],
             [-1.0, -1.0, -1.0, -1.0, -1.0], # a8 is missing
             [1.0, 2.0, 3.0, 4.0, 5.0],
             [1.0, 1.0, 1.0, 1.0, 1.0]]
        b = [-40.0, -2.0, -0.25, -4.0, -4.0, -1.0, -40.0, -60.0, 5.0, 1.0]

        # constraint
        g = []
        for j_ in range(5):
            somme1 = 0.0
            for i_ in range(5):
                somme1 = somme1 + -2.0*c[i_][j_]*X_[10+i_]
            somme2 = 0.0
            for i_ in range(10):
                somme2 = somme2 + a[i_][j_]*X_[i_]

            g.append(somme1 - e[j_] + somme2)

        f2 = g[0]
        f3 = g[1]

        w = []
        for i in range(len(g)):
            if g[i]>0.0:
                w.append(abs(g[i]))
            else:
                w.append(0.0)


        somme1 = 0.0
        somme2 = 0.0
        for j_ in range(5):
            somme2 = somme2 + 2.0*d[j_]*X_[10+j_]**3.0
            for i_ in range(5):
                somme1 = somme1 + c[i_][j_]*X_[10+j_]**2.0
        somme3 = 0.0
        for j_ in range(10):
            somme3 = somme3 - b[j_]*X_[j_]

        f1 = (somme1 + somme2 + somme3)
        f2 = (f2)
        f3 = (f3)

        f=[]                                                                        #add all the objective function value in a single list
        f.append(f1)
        f.append(f2)
        f.append(f3)


        for i in range(arg["Number of objectives"]):                                #send the original objective function values back
                    chromosome.objective_value.append(f[i])


        #stochastic ranking method
        if arg["Constraint method"] == "Sranking":
            sranking=Sranking(arg)
            sranking.sranking(objective_vector,chromosome,arg,w,f)

        #constraint dominancy method
        if arg["Constraint method"] == "Constraint_dominancy":
            constraint_dominancy=Constraint_dominancy(arg)
            constraint_dominancy.constraint_dominancy(objective_vector,chromosome,arg,w,f)


        #dynamic penalty method
        if arg["Constraint method"] =="Dynamic":
            penalty=dynamic(arg)
            penalty.dynamic_penalty(arg,objective_vector,chromosome,self.function_evaluation,w,f)


        # APM METHOD
        if arg["Constraint method"] == 'APM':
            apm=APM(arg)
            apm.apm_method(objective_vector,chromosome,arg,w,f)


        #self adaptive method
        if arg["Constraint method"]=='Sadaptive':
            sdaptive=Self_adaptive(arg)
            sdaptive.self_adaptive(objective_vector,chromosome,arg,w,f)


        return objective_vector

class G7_3D:
    def __init__(self, Nvar):
        Lbounds = []
        Ubounds = []
        del Lbounds[:]
        del Ubounds[:]
        for i in range(Nvar):
            Lbounds.append(float(-10.0))
            Ubounds.append(float(10.0))
        self.Lbounds = Lbounds
        self.Ubounds = Ubounds
        self.Name = "G7-3D"

        self.Bench_descret_matrix = {}
        self.address = "\\AMOEA_MAP_Package\\pareto_fronts_json\\G7_3D.json"
        self.function_evaluation = 0                                                    #this is an extra attribute added to implement dynamic penalty method

    def evaluate(self, chromosome, arg):
        objective_vector = []
        n_var = chromosome.Num_Var
        del objective_vector[:]

        X_ = []
        del X_[:]
        for i in range(n_var):
            X_.append(chromosome.variables[i])

        # constraint
        g = []
        g.append((4.0*X_[0] + 5.0*X_[1] - 3.0*X_[6] + 9.0*X_[7] - 105.0)/105.0)
        g.append((10.0*X_[0] - 8.0*X_[1] - 17.0*X_[6] + 2.0*X_[7])/370.0)
        g.append((-8.0*X_[0] + 2.0*X_[1] + 5.0*X_[8] -2.0*X_[9] - 12.0)/158.0)
        g.append((3.0*(X_[0]-2.0)**2.0 + 4.0*(X_[1]-3.0)**2.0 + 2.0*(X_[2])**2.0 - 7.0*X_[3] - 120.0)/1258.0)

        g.append((5.0*X_[0]**2.0 + 8.0*X_[1] + (X_[2]-6.0)**2.0 - 2.0*X_[3] - 40.0)/816.0)
        g.append((0.5*(X_[0]-8.0)**2.0 + 2.0*(X_[1]-4.0)**2.0 + 3.0*X_[4]**2.0 - X_[5] - 30.0)/834.0)
        g.append((X_[0]**2.0 + 2.0*(X_[1]-2.0)**2.0 - 2.0*X_[0]*X_[1] + 14.0*X_[4] - 6.0*X_[5])/788.0)
        g.append((-3.0*X_[0] + 6.0*X_[1] + 12.0*(X_[8]-8.0)**2.0 - 7.0*X_[9])/4048.0)

        w = []
        for i in range(len(g)):
            if g[i]>0.0:
                w.append(abs(g[i]))
            else:
                w.append(0.0)

        f_ = X_[0]**2.0 + X_[1]**2.0 + X_[0]*X_[1] - 14.0*X_[0] - 16.0*X_[1] + (X_[2]-10.0)**2.0 + 4.0*(X_[3]-5.0)**2.0
        f_ = f_ + (X_[4]-3.0)**2.0 + 2.0*(X_[5]-1.0)**2.0 + 5.0*X_[6]**2.0 + 7.0*(X_[7]-11.0)**2.0
        f1 = (f_)
        f2 = ((4.0*X_[0] + 5.0*X_[1] - 3.0*X_[6] + 9.0*X_[7] - 105.0)/105.0)
        f3 = ((10.0*X_[0] - 8.0*X_[1] - 17.0*X_[6] + 2.0*X_[7])/370.0)

        #appending the fitness values into a single list
        f=[]                                                                            #add all the objective function value in a single list
        f.append(f1)
        f.append(f2)
        f.append(f3)

        for i in range(arg["Number of objectives"]):
            chromosome.objective_value.append(f[i])


        #stochastic ranking method
        if arg["Constraint method"] == "Sranking":
            sranking=Sranking(arg)
            sranking.sranking(objective_vector,chromosome,arg,w,f)

        #constraint dominancy method
        if arg["Constraint method"] == "Constraint_dominancy":
            constraint_dominancy=Constraint_dominancy(arg)
            constraint_dominancy.constraint_dominancy(objective_vector,chromosome,arg,w,f)



        #dynamic penalty method
        if arg["Constraint method"] =="Dynamic":
            penalty=dynamic(arg)
            penalty.dynamic_penalty(arg,objective_vector,chromosome,self.function_evaluation,w,f)


        # APM METHOD
        if arg["Constraint method"] == 'APM':
            apm=APM(arg)
            apm.apm_method(objective_vector,chromosome,arg,w,f)


        #self adaptive method
        if arg["Constraint method"]=='Sadaptive':
            sdaptive=Self_adaptive(arg)
            sdaptive.self_adaptive(objective_vector,chromosome,arg,w,f)

        return objective_vector


class G7_2D:
    def __init__(self, Nvar):
        Lbounds = []
        Ubounds = []
        del Lbounds[:]
        del Ubounds[:]
        for i in range(Nvar):
            Lbounds.append(float(-10.0))
            Ubounds.append(float(10.0))
        self.Lbounds = Lbounds
        self.Ubounds = Ubounds
        self.Name = "G7-2D"

        self.Bench_descret_matrix = {}
        self.address = "\\AMOEA_MAP_Package\\pareto_fronts_json\\G7_2D.json"
        self.function_evaluations = 0                                                   #this is an extra attribute added to implement dynamic penalty method


    def evaluate(self, chromosome, arg):
        objective_vector = []
        n_var = chromosome.Num_Var
        del objective_vector[:]

        X_ = []
        del X_[:]
        for i in range(n_var):
            X_.append(chromosome.variables[i])

        # constraint
        g = []
        g.append((4.0*X_[0] + 5.0*X_[1] - 3.0*X_[6] + 9.0*X_[7] - 105.0)/105.0)
        g.append((10.0*X_[0] - 8.0*X_[1] - 17.0*X_[6] + 2.0*X_[7])/370.0)
        g.append((-8.0*X_[0] + 2.0*X_[1] + 5.0*X_[8] -2.0*X_[9] - 12.0)/158.0)
        g.append((3.0*(X_[0]-2.0)**2.0 + 4.0*(X_[1]-3.0)**2.0 + 2.0*(X_[2])**2.0 - 7.0*X_[3] - 120.0)/1258.0)

        g.append((5.0*X_[0]**2.0 + 8.0*X_[1] + (X_[2]-6.0)**2.0 - 2.0*X_[3] - 40.0)/816.0)
        g.append((0.5*(X_[0]-8.0)**2.0 + 2.0*(X_[1]-4.0)**2.0 + 3.0*X_[4]**2.0 - X_[5] - 30.0)/834.0)
        g.append((X_[0]**2.0 + 2.0*(X_[1]-2.0)**2.0 - 2.0*X_[0]*X_[1] + 14.0*X_[4] - 6.0*X_[5])/788.0)
        g.append((-3.0*X_[0] + 6.0*X_[1] + 12.0*(X_[8]-8.0)**2.0 - 7.0*X_[9])/4048.0)

        w = []
        for i in range(len(g)):
            if g[i]>0.0:
                w.append(abs(g[i]))
            else:
                w.append(0.0)
        constraint = sum(w)

        f_ = X_[0]**2.0 + X_[1]**2.0 + X_[0]*X_[1] - 14.0*X_[0] - 16.0*X_[1] + (X_[2]-10.0)**2.0 + 4.0*(X_[3]-5.0)**2.0
        f_ = f_ + (X_[4]-3.0)**2.0 + 2.0*(X_[5]-1.0)**2.0 + 5.0*X_[6]**2.0 + 7.0*(X_[7]-11.0)**2.0
        f1 = f_
        f2 = (4.0*X_[0] + 5.0*X_[1] - 3.0*X_[6] + 9.0*X_[7] - 105.0)/105.0

        #appending the fitness values into a single list
        f=[]                                                                            #add all the objective function value in a single list
        f.append(f1)
        f.append(f2)

        for i in range(arg["Number of objectives"]):
            chromosome.objective_value.append(f[i])


        #stochastic ranking method
        if arg["Constraint method"] == "Sranking":
            sranking=Sranking(arg)
            sranking.sranking(objective_vector,chromosome,arg,w,f)

        #constraint dominancy method
        if arg["Constraint method"] == "Constraint_dominancy":
            constraint_dominancy=Constraint_dominancy(arg)
            constraint_dominancy.constraint_dominancy(objective_vector,chromosome,arg,w,f)


        #dynamic penalty method
        if arg["Constraint method"] =="Dynamic":
            penalty=dynamic(arg)
            penalty.dynamic_penalty(arg,objective_vector,chromosome,self.function_evaluations,w,f)


        # APM METHOD
        if arg["Constraint method"] == 'APM':
            apm=APM(arg)
            apm.apm_method(objective_vector,chromosome,arg,w,f)


        #self adaptive method
        if arg["Constraint method"]=='Sadaptive':
            sdaptive=Self_adaptive(arg)
            sdaptive.self_adaptive(objective_vector,chromosome,arg,w,f)

        return objective_vector

class BICOP1_2D:
    def __init__(self, Nvar):
        Lbounds = []
        Ubounds = []
        del Lbounds[:]
        del Ubounds[:]
        for i in range(Nvar):
            Lbounds.append(float(0.0))
            Ubounds.append(float(1.0))
        self.Lbounds = Lbounds
        self.Ubounds = Ubounds
        self.Name = "BICOP1"
        self.Bench_descret_matrix = {}
        self.address = "\\AMOEA_MAP_Package\\pareto_fronts_json\\BICOP1_2D.json"
        self.function_evaluations = 0                                                   #this is an extra attribute added to implement dynamic penalty method


    def evaluate(self, chromosome, arg):
        objective_vector = []
        n_var = chromosome.Num_Var
        del objective_vector[:]

        X_ = []
        del X_[:]
        for i in range(n_var):
            X_.append(chromosome.variables[i])

        # constraint
        g = []
        somme = 0
        for i in range(n_var-1):
            somme = somme + X_[i+1]/(float(n_var)-1.0)
        g.append(-(1.0+9.0*somme))

        w = []
        for i in range(len(g)):
            if g[i]>0.0:
                w.append(abs(g[i]))
            else:
                w.append(0.0)


        f1 = somme*X_[0]
        if somme!=0:
            f2 = somme*(1.0-(X_[0]/somme)**0.5)
        else:
            f2 = 0

       #appending the fitness values into a single list
        f=[]                                                                                #add all the objective function value in a single list
        f.append(f1)
        f.append(f2)



        for i in range(arg["Number of objectives"]):
                    chromosome.objective_value.append(f[i])

        #stochastic ranking method
        if arg["Constraint method"] == "Sranking":
            sranking=Sranking(arg)
            sranking.sranking(objective_vector,chromosome,arg,w,f)

        #constraint dominancy method
        if arg["Constraint method"] == "Constraint_dominancy":
            constraint_dominancy=Constraint_dominancy(arg)
            constraint_dominancy.constraint_dominancy(objective_vector,chromosome,arg,w,f)


        #dynamic penalty method
        if arg["Constraint method"] =="Dynamic":
            penalty=dynamic(arg)
            penalty.dynamic_penalty(arg,objective_vector,chromosome,self.function_evaluations,w,f)


        # APM METHOD
        if arg["Constraint method"] == 'APM':
            apm=APM(arg)
            apm.apm_method(objective_vector,chromosome,arg,w,f)


        #self adaptive method
        if arg["Constraint method"]=='Sadaptive':
            sdaptive=Self_adaptive(arg)
            sdaptive.self_adaptive(objective_vector,chromosome,arg,w,f)


        return objective_vector

class OSY_2D:
    def __init__(self, Nvar):
        Lbounds = []
        Ubounds = []
        del Lbounds[:]
        del Ubounds[:]
        if Nvar<>6: print "number of variables must be 6"
        Lbounds.append(float(0.0))
        Lbounds.append(float(0.0))
        Lbounds.append(float(1.0))
        Lbounds.append(float(0.0))
        Lbounds.append(float(1.0))
        Lbounds.append(float(0.0))
        Ubounds.append(float(10.0))
        Ubounds.append(float(10.0))
        Ubounds.append(float(5.0))
        Ubounds.append(float(6.0))
        Ubounds.append(float(5.0))
        Ubounds.append(float(10.0))

        self.Lbounds = Lbounds
        self.Ubounds = Ubounds
        self.Name = "OSY"
        self.Bench_descret_matrix = {}
        self.address = "\\AMOEA_MAP_Package\\pareto_fronts_json\\OSY_2D.json"
        self.function_evaluations = 0                                                       #this is an extra attribute added to implement dynamic penalty method

    def evaluate(self, chromosome, arg):
        objective_vector = []
        n_var = chromosome.Num_Var
        del objective_vector[:]

        X_ = []
        del X_[:]
        for i in range(n_var):
            X_.append(chromosome.variables[i])

        f1 = -(25.0*(X_[0]-2.0)**2 + (X_[1]-2.0)**2 + (X_[2]-1.0)**2 + (X_[3]-4.0)**2 + (X_[4]-1.0)**2)
        f2 = X_[0]**2 + X_[1]**2 + X_[2]**2 + X_[3]**2 + X_[4]**2 + X_[5]**2

        # constraints
        g = []
        g.append(2.0-X_[0]-X_[1]) # C1
        g.append(X_[0]+X_[1]-6.0) # C2
        g.append(-X_[0]+X_[1]-2.0) # C3
        g.append(X_[0]-3.0*X_[1]-2.0) # C4
        g.append(X_[3]+(X_[2]-3.0)**2-4.0) # C5
        g.append(-X_[5]-(X_[4]-3.0)**2+4.0) # C6

        w = []
        for i in range(6):
            if g[i]>0.0:
                w.append(abs(g[i]))
            else:
                w.append(0.0)

       #appending the fitness values into a single list
        f=[]                                                                            #add all the objective function value in a single list
        f.append(f1)
        f.append(f2)


        for i in range(arg["Number of objectives"]):
                    chromosome.objective_value.append(f[i])


        #stochastic ranking method
        if arg["Constraint method"] == "Sranking":
            sranking=Sranking(arg)
            sranking.sranking(objective_vector,chromosome,arg,w,f)

        #constraint dominancy method
        if arg["Constraint method"] == "Constraint_dominancy":
            constraint_dominancy=Constraint_dominancy(arg)
            constraint_dominancy.constraint_dominancy(objective_vector,chromosome,arg,w,f)



        #dynamic penalty method
        if arg["Constraint method"] =="Dynamic":
            penalty=dynamic(arg)
            penalty.dynamic_penalty(arg,objective_vector,chromosome,self.function_evaluations,w,f)


        # APM METHOD
        if arg["Constraint method"] == 'APM':
            apm=APM(arg)
            apm.apm_method(objective_vector,chromosome,arg,w,f)


        #self adaptive method
        if arg["Constraint method"]=='Sadaptive':
            sdaptive=Self_adaptive(arg)
            sdaptive.self_adaptive(objective_vector,chromosome,arg,w,f)


        return objective_vector




class beam_design_2D:
    # The welded beam design problem 2D
    # Deb, Sundar, Rao KanGAL Report Number 2005012
    def __init__(self, Nvar):
        Lbounds = []
        Ubounds = []
        del Lbounds[:]
        del Ubounds[:]
        Lbounds.append(float(0.125))    # h
        Lbounds.append(float(0.1))      # l
        Lbounds.append(float(0.1))      # t
        Lbounds.append(float(0.125))    # b
        Ubounds.append(float(5.0))
        Ubounds.append(float(10.0))
        Ubounds.append(float(10.0))
        Ubounds.append(float(5.0))
        self.Lbounds = Lbounds
        self.Ubounds = Ubounds
        self.Name = "Beam design"
        self.Bench_descret_matrix = {}
        self.address = "\\AMOEA_MAP_Package\\pareto_fronts_json\\beam_design_2D.json"
        self.function_evaluations = 0                                                           #this is an extra attribute added to implement dynamic penalty method

    def evaluate(self, chromosome,arg):    #############
        objective_vector = []
        n_var = chromosome.Num_Var
        del objective_vector[:]
        X_ = []
        del X_[:]
        for i in range(n_var):
            X_.append(chromosome.variables[i])

        h = X_[0]
        l = X_[1]
        t = X_[2]
        b = X_[3]

        f1 = 1.10471*(h**2.0)*l+0.04811*t*b*(14.0+l)

        Tau1 = 6000.0/(math.sqrt(2.0)*h*l)
        Tau2 = 6000.0*(14.0+0.5*l)*math.sqrt(0.25*(l**2.0+(h+t)**2.0)) / (2.0*(0.707*h*l)*((l**2.0)/12.0+0.25*(h+t)**2.0))
        sigma = 504000.0/((t**2.0)*b)
        Pc = 64746.022*(1.0-0.0282346*t)*t*(b**3.0)

        f2 = 2.1952/((t**3.0)*b)

        Tau = math.sqrt(Tau1**2.0+Tau2**2.0+l*Tau1*Tau2/math.sqrt(0.25*(l**2.0+(h+t)**2.0)))

        # constraints
        g = []
        p1 = 100
        p2 = 0.1
        g.append((-Tau+13600.0)*p1)
        g.append((-sigma+30000.0)*p2)
        g.append(-h+b)
        g.append(Pc-6000.0)
        #g.append(-delta+0.25)            #why this constraint is not incluced

        #calculating the absolute value of the constraint violation
        w = []
        for i in range(arg["Number of constraints"]):
            if g[i]<0.0:
                w.append(abs(g[i]))
            else:
                w.append(0.0)

        #appending the fitness values into a single list
        f=[]                                                                                    #add all the objective function value in a single list
        f.append(f1)
        f.append(f2)

        for i in range(arg["Number of objectives"]):
            chromosome.objective_value.append(f[i])

        #stochastic ranking method
        if arg["Constraint method"] == "Sranking":
            sranking=Sranking(arg)
            sranking.sranking(objective_vector,chromosome,arg,w,f)

        #constraint dominancy method
        if arg["Constraint method"] == "Constraint_dominancy":
            constraint_dominancy=Constraint_dominancy(arg)
            constraint_dominancy.constraint_dominancy(objective_vector,chromosome,arg,w,f)



        #dynamic penalty method
        if arg["Constraint method"] =="Dynamic":
            penalty=dynamic(arg)
            penalty.dynamic_penalty(arg,objective_vector,chromosome,self.function_evaluations,w,f)


        # APM METHOD
        if arg["Constraint method"] == 'APM':
            apm=APM(arg)
            apm.apm_method(objective_vector,chromosome,arg,w,f)


        #self adaptive method
        if arg["Constraint method"]=='Sadaptive':
            sdaptive=Self_adaptive(arg)
            sdaptive.self_adaptive(objective_vector,chromosome,arg,w,f)


        return objective_vector

####################################################################################################################################################################################################


#class for calculating dynamic penalty


"""
this class will use number of function evaluations in the place of generation number to apply penalty on the
individuals.In each generation number of function evaluations will increase so the penalty applied will be
higher with each generation.(As proposed by the method to increase the penalty factor with each generation)
"""
class dynamic:

    def __init__(self,arg):
        self.arg=arg

    def dynamic_penalty(self,arg,objective_vector,chromosome,function_evaluations,w,f):
        for i in range(arg["Number of constraints"]):
            chromosome.constraint_value.append(w[i])
        time=float(function_evaluations)
        Di=(sum(w)**1)*(0.5*time)**2
        for i in range(arg["Number of objectives"]):
            objective_vector.append(f[i]+Di)
        chromosome.constraint = sum(w)





"""
This class will implement the penalty adaptively.The penalty factor will be based on
average objective function value and average constraint violation of the last generation.
"""

class APM:
    def __init__(self,arg):
        self.arg=arg

    def apm_method(self,objective_vector,chromosome,arg,w,f):
        for i in range(arg["Number of constraints"]):
            chromosome.constraint_value.append(w[i])
        if sum(w)==0:                                                   #if the individual is feasible the there will be no penalty
                for i in range(arg["Number of objectives"]):
                    objective_vector.append(f[i])
                chromosome.constraint=0

        else:
                Penalty_factor=[]                                                #here is the penalty factor calculation based on average objective function value and average constraint violation

                div=sum(x*x for x,x in zip(arg["Average values"][arg["Number of objectives"]:],arg["Average values"][arg["Number of objectives"]:]))

                for i in range(arg["Number of objectives"]):
                    k1=[]
                    for t_ in range(arg["Number of constraints"]):
                        if div!=0:
                            k1.append((arg["Average values"][i])*(arg["Average values"][arg["Number of objectives"]+t_])/div)
                        else:
                            k1.append(0)
                    Penalty_factor.append(k1)



                for i in range(arg["Number of objectives"]):                        #implication of penalty on the individuals
                    if f[i] > arg["Average values"][i]:
                        objective_vector.append(f[i]+sum(k1*w for k1,w in zip(Penalty_factor[0],w)))
                    else:
                        somme=arg["Average values"][i]+sum(k1*w for k1,w in zip(Penalty_factor[0],w))
                        objective_vector.append(somme)

                chromosome.constraint=sum(w)





"""
this is class for self adaptive method.Here the penalty is applied in two stages
and it is based on the best , the worst and the highest objective value individual.
"""

class Self_adaptive:
    def __init__(self,arg):
        self.arg=arg

    def self_adaptive(self,objective_vector,chromosome,arg,w,f):
        for i in range(arg["Number of constraints"]):
                chromosome.constraint_value.append(w[i])

        Infeasibility_measure=0                                                             #calculate infeasibility measure for current individual based on the max constraint
        for i in range(arg["Number of constraints"]):
            if arg["max_constraint"][i]!=0:
                Infeasibility_measure += w[i]/arg["max_constraint"][i]
        Infeasibility_measure=Infeasibility_measure/arg["Number of constraints"]


        First_stage_penalty=[]                                              #calculate first stage penalty and it is applied if there is no feasible solution in the population
        if not arg["any_feasible"]:
            for i in range(1,2*arg["Number of objectives"],2):
                if (arg["worst"][i]-arg["best"][i]) == 0:
                    First_stage_penalty.append(0)
                else:
                    First_stage_penalty.append((Infeasibility_measure-arg["best"][i])/(arg["worst"][i]-arg["best"][i]))
        else:
            for i in range(1,2*arg["Number of objectives"],2):
                First_stage_penalty.append(0.0)



        for i in range(arg["Number of objectives"]):                #limiting the penalty factor to math range
            if First_stage_penalty[i]>354.5:
                First_stage_penalty[i]=354.5


        f_dot=[]
        for i in range(arg["Number of objectives"]):
            f_dot.append(f[i]+First_stage_penalty[i]*(arg["best"][2*i]-arg["worst"][2*i]))      #here the first penalty is applied to the objective function




            #now 2nd exponential penalty will be applied to the objective function


        gamma=[]                                                            #gamma function suggested by the authors
        for i in range(1,2*arg["Number of objectives"],2):
            if arg["worst"][i-1] <= arg["best"][i-1]:
                gamma.append((arg["highest"][i/2]-arg["best"][i-1])/arg["best"][i-1])
            if arg["worst"][i-1] == arg["highest"][i/2]:
                gamma.append(0)
            if arg["worst"][i-1] > arg["best"][i-1]:
                if arg["worst"][i-1]==0:
                    arg["worst"][i-1]=arg["best"][i-1]
                gamma.append((arg["highest"][i/2]-arg["worst"][i-1])/arg["worst"][i-1])

        Second_stage_penalty=[]
        for i in range(arg["Number of objectives"]):
            Second_stage_penalty.append(gamma[i]*f_dot[i]*((math.exp(2.0*First_stage_penalty[i])-1)/(math.exp(2.0)-1)))


        for i in range(arg["Number of objectives"]):                                            #here second exponential penalty is applied
            objective_vector.append(f_dot[i]+Second_stage_penalty[i])

        chromosome.constraint=sum(w)




"""
this is the class for stochastic ranking method.
"""


class Sranking:
    def __init__(self,arg):
        self.arg=arg

    def sranking(self,objective_vector,chromosome,arg,w,f):
        for i in range(arg["Number of constraints"]):
                chromosome.constraint_value.append(w[i])
        constraint=sum(w)
        for i in range(arg["Number of objectives"]):
            objective_vector.append(f[i])
        chromosome.constraint=constraint

"""
this is the class for the constraint dominancy method.
"""

class Constraint_dominancy:
    def __init__(self,arg):
        self.arg=arg

    def constraint_dominancy(self,objective_vector,chromosome,arg,w,f):
        for i in range(arg["Number of constraints"]):
                chromosome.constraint_value.append(w[i])
        constraint=sum(w)
        for i in range(arg["Number of objectives"]):
            objective_vector.append(f[i])
        chromosome.constraint=constraint


"""
this is the class for alpha constraint method.It is a single objective constraint handling method.
"""

class alpha_constraint:
    def __init__(self,arg):
        self.arg=arg

    def alpha_constraint(self,objective_vector,chromosome,arg,w,f):
        for i in range(len(w)):
            chromosome.constraint_value.append(w[i])
        objective_vector.append(f)
        chromosome.constraint=sum(w)

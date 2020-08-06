#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import networkx as nx
import operator
import time
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu


# In[2]:



#best=21294
x_c=[2995 , 202  , 981  , 1346 , 781  , 1009 , 2927 , 2982 , 555  , 464  , 3452 , 571  , 2656 , 1623 , 2067 , 1725 , 
   3600 , 1109 , 366  , 778  , 386  , 3918 , 3332 , 2597 , 811  , 241  , 2658 , 394  , 3786 , 264  , 2050 , 3538 ,
   1646 , 2993 , 547  , 3373 , 460  , 3060 , 1828 , 1021 , 2347 , 3535 , 1529 , 1203 , 1787 , 2740 , 555  , 47   ,
   3935 , 3062 , 387  , 2901 , 931  , 1766 , 401  , 149  , 2214 , 3805 , 1179 , 1017 , 2834 , 634  , 1819 , 1393 ,
   1768 , 3023 , 3248 , 1632 , 2223 , 3868 , 1541 , 2374 , 1962 , 3007 , 3220 , 2356 , 1604 , 2028 , 2581 , 2221 ,
   2944 , 1082 , 997  , 2334 , 1264 , 1699 , 235  , 2592 , 3642 , 3599 , 1766 , 240  , 1272 , 3503 , 80   , 1677 ,
   3766 , 3946 , 1994 ,  278  ]
y_c=[264 , 233 , 848 , 408 , 670 , 1001 , 1777 , 949 , 1121 , 1302 , 637 , 1982 , 128 , 1723 , 694 , 927 , 459 , 1196 ,
   339 , 1282 , 1616 , 1217 , 1049 , 349 , 1295 , 1069 , 360 , 1944 , 1862 , 36 , 1833 , 125 , 1817 , 624 , 25 , 1902 ,
   267 , 781 , 456 , 962 , 388 , 1112 , 581 , 385 , 1902 , 1101 , 1753 , 363 , 540 , 329 , 199 , 920 , 512 , 692 , 980 ,
   1629 , 1977 , 1619 , 969 , 333 , 1512 , 294 , 814 , 859 , 1578 , 871 , 1906 , 1742 , 990 , 697 , 354 , 1944 , 389 ,
   1524 , 1945 , 1568 , 706 , 1736 , 121 , 1578 , 632 , 1561 , 942 , 523 , 1090 , 1294 , 1059 , 248 , 699 , 514 , 678 , 
   619 , 246 , 301 , 1533 , 1238 , 154 , 459 , 1852 , 165 ]
print("Data Size\nX: ",len(x_c),"\tY: ",len(y_c))


# In[3]:


N = len(x_c)
def objFunc(p):
    distance = 0
    for i in range(N - 1):
        xd = x_c[p[i]] - x_c[p[i+1]]
        yd = y_c[p[i]] - y_c[p[i+1]]
        
        dxy = np.sqrt( (xd * xd) + (yd * yd))
        distance = distance + dxy
        
    xd = x_c[p[N-1]] - x_c[p[0]]
    yd = y_c[p[N-1]] - y_c[p[0]]
        
    dxy = np.sqrt( (xd * xd) + (yd * yd))
    distance = distance + dxy
    
    return distance


# # ABC parameters initialization

# In[4]:


SN = 30 # Colony Size
FoodNum = int (SN/2) # Number of foods
limit = FoodNum * N # Parameter limit
MaxNFE = 1000000 # Maximum number of func. evaluations
num_eval = 0 # Number of func. evaluations

Foods = np.zeros((FoodNum,N), dtype=int)#food for each bee
trials = np.zeros(FoodNum, dtype=int)
Fitness = np.zeros(FoodNum, dtype=float)
GlobalMin = np.finfo('f').max
GlobalParams = np.zeros(N, dtype=int) # best global tour
Prob = np.zeros(FoodNum, dtype=float)


# # Swapping functions

# ## Two Opt

# In[5]:


def TwoOpt(p):
    r1 = np.random.randint(N-1)
    r2 = np.random.randint(N-1)
    while r1 == r2:
        r2 = np.random.randint(N-1)
        
    if r1 > r2:
        temp = r1
        r1 = r2
        r2 = temp
        
    new_p = p[:r1].copy()
    new_p = np.append(new_p, np.flip(p[r1:r2]))
    new_p = np.append(new_p , p[r2:])
    
    return new_p


# ## IPMX

# In[6]:


def edit_exchange_list_func(start,tail,edited_exchange_list):
    for column in range(2):
        a=edited_exchange_list[:,column]
        for i,j in zip(a,range(len(a))):
            if(i==start):
                if column==1:
                    edited_exchange_list[j,0]=tail
                else:
                    edited_exchange_list[j,1]=tail
                
    return edited_exchange_list
def check_same_tuples(G,component):
    checker=False
    for c in component:
        if(len(G.out_edges(c))==1 and len(G.in_edges(c))==1):
            checker=True
        else:
            checker=False
    return checker

def IPMX(p1):
    p2= np.random.permutation(N)
    #p2_random = np.random.randint(FoodNum)
    #p2=Foods[p2_random]

    #steps 1 & 2
    offspring1=p1.copy()
    offspring2=p2.copy()
    
    r1 = np.random.randint(N-1)
    r2 = np.random.randint(N-1)
    while r1 == r2:
        r2 = np.random.randint(N-1)   
    if r1 > r2:
        temp = r1
        r1 = r2
        r2 = temp
    temp = p1.copy()
    offspring1[r1:r2] = p2[r1:r2]
    offspring2[r1:r2] = temp[r1:r2]
    
    #steps 3 & 4
    exchange_list_length=r2-r1
    exchange_list = np.zeros((exchange_list_length, 3),dtype='int')
    for i in range(r1,r2):
        
        exchange_list[i-r1][0]=offspring1[i]
        exchange_list[i-r1][1]=offspring2[i]
        exchange_list[i-r1][2]=1
    #print("exchange_list")
    #print(exchange_list)
    
    
    #steps 5,6,& 7
    guide_list=[-1 for i in range(N)]
    guide_list=np.asarray(guide_list,dtype='int')
    
    guide_list[exchange_list[:,0]]=exchange_list[:,1]
    #print("guide_list")
    #print(guide_list)
    l1=[-1 for i in range(N)]
    l1=np.asarray(l1,dtype='int') 
    l2=[-1 for i in range(N)]
    l2=np.asarray(l2,dtype='int')
    l1[exchange_list[:,0]]=1
    l2[exchange_list[:,1]]=1
    #print("l1: ",l1)
    #print("l2: ",l2)
    sum_l=np.add(l1,l2)
    #print("sum:  ")
    #print(sum_l)

    
    plt.clf()
    #create graph
    G=nx.DiGraph()
    exchange_list_flatten = exchange_list[:,:-1].flatten('F')
    # add nodes
    G.add_nodes_from(exchange_list_flatten)
    
    exchange_list_tuple=[tuple(e) for e in exchange_list[:,:-1] ]
    
    #add edges
    exchange_list_edges=list(zip(exchange_list[:,0], exchange_list[:,1])) 
    G.add_edges_from(exchange_list_tuple)

    #check number of nodes in each component
    node_subset=[i for i in range(N)]
    undirected_G=G.to_undirected()
    components= [list(cc) for cc in nx.connected_components(undirected_G)]
    #print("components")
    #print(components)
    
    edited_exchange_list=exchange_list.copy()
    starts=[]
    mirrors_component=[]
    same_nodes=[]
    for component in components:
        if(len(component)>2):
            start=-1
            tail=-1
            #print(component)
            for c in component:
                if len(G.in_edges(c))==0:
                    start=c
                    starts.append(start)
                if len(G.out_edges(c))==0:
                    tail=c
            if start!= -1 and tail!= -1:
                edited_exchange_list=edit_exchange_list_func(start,tail,edited_exchange_list)        
                #print("add edge between: ",start,"   ",tail)
                G.add_edge(start,tail)
        if(len(component)==1):
            same_nodes.append(component[0])
    """
    #print("updated exchange list")
    #print(edited_exchange_list)
     #plotting the grph 
   
    pos = nx.spring_layout(G)
    # Draw the graph according to node positions
    nx.draw(G, pos, with_labels=True)

    # Create edge labels
    labels = nx.get_edge_attributes(G,'weight')

    # Draw edge labels according to node positions
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
   """
    #plt.show()
    
    #print("same:            ",same_nodes)
    #print("starts:     ",starts)
    #print("mirrors:        \n",mirrors_component)
    for i,j in zip(sum_l,range(len(sum_l))):            
        if i == 2:
            index=np.where(exchange_list[:,0] == j)
            #print("to check:    ",edited_exchange_list[index][0][:-1])
            check = any(item in starts for item in edited_exchange_list[index][0][:-1])
            #print("check : ",check)
            same = np.all(edited_exchange_list[index][0][:-1]==edited_exchange_list[index][0][0])
            #print("same:   ",same)
            if (not check and not same):
                edited_exchange_list[index[0],2]=0
    
    #print(edited_exchange_list)
    for i,j in zip(edited_exchange_list[:,2],range(len(edited_exchange_list[:,2]))):
        #print(i,"    ",j)
        if i == 1:
            guide_list[edited_exchange_list[j,0]]=edited_exchange_list[j,1]
        else:
            guide_list[edited_exchange_list[j,0]] = -1
    #print("after guide_list")
    #print(guide_list)
    
    
    #setp 8
    for i,j in zip(guide_list,range(len(guide_list))):
        
        if i != -1:
            index=offspring1.tolist().index(j)
            offspring1[index]= i

    #print(len(set(offspring1)))
    
    
    #steps 9 && 10
    f=np.zeros((N),dtype='int')
    f[offspring1]=p1
    offspring2[:]=f[p2]
    #print(len(set(offspring2)))
    #print("offspring1:\n",offspring1)
    #print("offspring2:\n",offspring2)
    G.clear()
    
    return offspring1


# # ABC

# In[7]:


# Fitness function
def CalculateFitness(p):
    objval = objFunc(p)
    result=0
    if objval>=0:	 
            result=1/(objval+1)
    else:
            result=1+np.fabs(objval)
    return result



def init(k):
    Foods[k][:] = np.random.permutation(N)
    Fitness[k] = objFunc(Foods[k])
    global num_eval 
    num_eval = num_eval + 1
    trials[k] = 0



def Initialization():
    for i in range(FoodNum):
        init(i)

def EmployedBees(method):
    #global iteration
    for i in range(FoodNum):
        if method ==1:
            New_Solution = TwoOpt(Foods[i])
        else:
            New_Solution = IPMX(Foods[i])
        New_Fitness = objFunc(New_Solution)
        global num_eval 
        #global fitnesses
        num_eval = num_eval + 1
        
        #print("EmployedBees: ",num_eval)
        
        if New_Fitness < Fitness[i]:
            Fitness[i] = New_Fitness
            best=New_Fitness
            Foods[i][:] = New_Solution
            trials[i] = 0
        else:
            trials[i] = trials[i] +1
            best=Fitness[i]
    #fitnesses[iteration]=best
    #print(iteration)
    #iteration+=1

def CalculateProbabilities():
    total = 0
    for i in range(FoodNum):
        total = total + (1/Fitness[i])
        
    for i in range(FoodNum):
        Prob[i] = (1/Fitness[i]) / total
        

  
def OnlookerBees(method):
    t = 0
    k = 0
    #global iteration
    while t < FoodNum:
        r = np.random.rand()
        
        if r < Prob[k]:
            t = t + 1
            if method == 1:
                New_Solution = TwoOpt(Foods[k])
            else:
                New_Solution = IPMX(Foods[k])
            New_Fitness = objFunc(New_Solution)
            best = objFunc(New_Solution)
            global num_eval 
            #global fitnesses
            num_eval = num_eval + 1
            #print("OnlookerBees: ",num_eval)

            if New_Fitness < Fitness[k]:
                Fitness[k] = New_Fitness
                Foods[k][:] = New_Solution
                best = New_Fitness
                trials[k] = 0
            else:
                trials[k] = trials[k] +1
                best =Fitness[k]
            #fitnesses[iteration]=best
            #print(iteration)
            #iteration+=1
        k = k + 1        
        if k == FoodNum:
            k = 0
        

def SendScoutBees():
    max_index = np.argmax(trials, axis=0)
    if trials[max_index] >= limit:
        init(max_index)
        #print("Trial:",max_index)
    
  
def MemorizeBestSource():
    global GlobalMin, GlobalParams
    if np.min(Fitness) < GlobalMin:
        GlobalMin = np.min(Fitness)
        GlobalParams = Foods[np.argmin(Fitness, axis=0)][:]

       

def ABC(method):
 
    Initialization()
    MemorizeBestSource()
    global num_eval
    best=0
    fitnesses=np.zeros(MaxNFE)
    num_eval=0
    prev_num_eval=0
      
    while num_eval < MaxNFE:
        prev_num_eval=num_eval
        EmployedBees(method)
        if num_eval >= MaxNFE:
            break
        CalculateProbabilities()
        OnlookerBees(method)
        if num_eval >= MaxNFE:
            break
        MemorizeBestSource()
        SendScoutBees()
        global GlobalMin
        best=GlobalMin
    #fitnesses[MaxNFE-1]=GlobalMin
       # print("Fitness: ", GlobalMin)
    #print("Best: ", objFunc(GlobalParams))
        for i in range(prev_num_eval,num_eval):
            fitnesses[i]=GlobalMin
            
    for i in range(prev_num_eval,num_eval):
        if i < MaxNFE:
            fitnesses[i]=best
        else:
            fitnesses[MaxNFE-1]=best
            
    return best,fitnesses


# In[8]:


#ABC()


# # ACO

# ## classes of Graph , ACS , Ant

# In[9]:


# rank = number_of_cities
class Graph(object):
    def __init__(self, distances: list, rank: int):
        self.distances = distances
        self.rank = rank
        self.pheromone = [[1 / (rank * rank) for j in range(rank)] for i in range(rank)]
class ACS(object):
    def __init__(self, ant_count: int, generations: int, alpha: float, beta: float, rho: float, q: int):
        self.generations = generations
        self.ant_count = ant_count
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = q
class Ant(object):
    def __init__(self, acs: ACS, graph: Graph):
        self.colony = acs
        self.graph = graph
        self.total_cost = 0.0
        self.visited_nodes = []
        self.pheromone_delta = []
        self.unvisited_nodes = [i for i in range(graph.rank)]
        start_node = random.randint(0, graph.rank - 1)
        self.visited_nodes.append(start_node)
        self.current_node = start_node
        self.unvisited_nodes.remove(start_node)
    def re_init(self, graph: Graph):
        self.graph = graph
        self.total_cost = 0.0
        self.visited_nodes = []
        self.pheromone_delta = []
        #self.unvisited_nodes = [i for i in range(graph.rank)]


# ## update pheromone ,  update pheromone delta & solve functions

# In[10]:


def update_pheromone(self, graph: Graph, ants: list):
        for i, row in enumerate(graph.pheromone):
            for j, col in enumerate(row):
                sum_ants_pheromone_delta = 0
                for ant in ants:
                    sum_ants_pheromone_delta += ant.pheromone_delta[i][j]
                graph.pheromone[i][j] = (1 - self.rho) * graph.pheromone[i][j] + sum_ants_pheromone_delta
                
ACS.update_pheromone = update_pheromone

def all_same(items):
     return all(x == items[0] for x in items)

def solve(self, graph: Graph,method:int):
        best_cost = float('inf')
        best_solution = []
        fitnesses=np.zeros(self.generations)
        for _ in range(self.generations):
            ants = [Ant(self, graph) for i in range(self.ant_count)]
            #print(ants)
            for ant in ants:
                #ant.re_init(graph)
                for i in range(graph.rank - 1):
                    ant.select_next_node()
                if method == 1: 
                    ant.two_opt(graph)
                else:
                    ant.ipmx(graph)
                #ant.total_cost += graph.distances[ant.visited_nodes[-1]][ant.visited_nodes[0]]
                if ant.total_cost < best_cost:
                    best_cost = ant.total_cost
                    best_solution = [] + ant.visited_nodes
                ant.update_pheromone_delta()
                #print(ant.visited_nodes)
            fitnesses[_]=best_cost
            self.update_pheromone(graph, ants)
            #print(_,"       best : ",best_cost)
        return best_cost, fitnesses
    
ACS.solve = solve

def update_pheromone_delta(self):
        self.pheromone_delta = [[0 for j in range(self.graph.rank)] for i in range(self.graph.rank)]
        for _ in range(1, len(self.visited_nodes)):
            i = self.visited_nodes[_ - 1]
            j = self.visited_nodes[_]
            self.pheromone_delta[i][j] = self.colony.Q / self.total_cost
            
Ant.update_pheromone_delta = update_pheromone_delta


# ## 2-opt & IPMX neighborhood operators 

# In[11]:


def two_opt(self,graph:Graph):
    tour=self.visited_nodes
    #print(tour)
    #print("tttttttttttt: \n",tour)
    #print("MMMMMMMMMMMMMMMMMMMM: ",len(tour))
    #print("datatype: ",type(tour))
    #print("dddddddddddddddddddd, ",self.visited_nodes)
    #self.re_init(graph)
    N=len(tour)
    self.graph = graph
    self.total_cost = 0.0
    self.visited_nodes = []
    self.pheromone_delta = []
    self.unvisited_nodes = [i for i in range(graph.rank)]
    r1 = np.random.randint(N)
    r2 = np.random.randint(N)
    while r1 == r2:
        r2 = np.random.randint(N)
        
    if r1 > r2:
        temp = r1
        r1 = r2
        r2 = temp
    new_p=tour.copy()
    new_p[r1:r2+1]=np.flip(tour[r1:r2+1])

    
    #print(new_p)
    for i in range(N-1):
        self.visited_nodes.append(new_p[i])
        self.total_cost += self.graph.distances[new_p[i]][new_p[i+1]]
    self.visited_nodes.append(new_p[N-1])
    self.total_cost += self.graph.distances[new_p[N-1]][new_p[0]]
    
    #print("new_p   : ",self.visited_nodes)
Ant.two_opt = two_opt

def select_next_node(self):
    denominator = 0
    for i in self.unvisited_nodes:
        denominator += self.graph.pheromone[self.current_node][i] ** self.colony.alpha *             self.graph.distances[self.current_node][i] ** (-1 * self.colony.beta )
    #print("denominator: ",denominator)
    if denominator ==0 :
        denominator= random.uniform(0,2)
    probabilities = [0 for i in range(self.graph.rank)]
    for i in range(self.graph.rank):
        try:
            self.unvisited_nodes.index(i)  # test if allowed list contains i
            probabilities[i] = self.graph.pheromone[self.current_node][i] ** self.colony.alpha *                 self.graph.distances[self.current_node][i] ** (-1 * self.colony.beta) / denominator
        except ValueError:
            pass

    # select next node by probability roulette
    selected_node = 0
    rand = random.random()
    for i, probability in enumerate(probabilities):
        rand -= probability
        if rand <= 0:
            selected_node = i
            break

    try:
        self.unvisited_nodes.remove(selected_node)
    except ValueError:
        pass
    
    self.visited_nodes.append(selected_node)
    self.total_cost += self.graph.distances[self.current_node][selected_node]
    self.current_node = selected_node
    #print(self.current_node )
    
Ant.select_next_node = select_next_node



def edit_exchange_list_func(start,tail,edited_exchange_list):
    for column in range(2):
        a=edited_exchange_list[:,column]
        for i,j in zip(a,range(len(a))):
            if(i==start):
                if column==1:
                    edited_exchange_list[j,0]=tail
                else:
                    edited_exchange_list[j,1]=tail
                
    return edited_exchange_list
def check_same_tuples(G,component):
    checker=False
    for c in component:
        if(len(G.out_edges(c))==1 and len(G.in_edges(c))==1):
            checker=True
        else:
            checker=False
    return checker
def ipmx(self,graph:Graph):
    #print("jhjjj")
    p1=self.visited_nodes
    #print(self.visited_nodes)
    N = len(p1)
    p2= np.random.permutation(N)
    
    self.re_init(graph)
    
    #steps 1 & 2
    offspring1=p1.copy()
    offspring2=p2.copy()
    
    r1 = np.random.randint(N-20)
    r2 = np.random.randint(N)
    while r1 == r2:
        r2 = np.random.randint(N)   
    if r1 > r2:
        temp = r1
        r1 = r2
        r2 = temp
    temp = p1.copy()
    offspring1[r1:r2] = p2[r1:r2]
    offspring2[r1:r2] = temp[r1:r2]
    
    #steps 3 & 4
    exchange_list_length=r2-r1
    exchange_list = np.zeros((exchange_list_length, 3),dtype='int')
    for i in range(r1,r2):
        
        exchange_list[i-r1][0]=offspring1[i]
        exchange_list[i-r1][1]=offspring2[i]
        exchange_list[i-r1][2]=1
    #print("exchange_list")
    #print(exchange_list)
    
    
    #steps 5,6,& 7
    guide_list=[-1 for i in range(N)]
    guide_list=np.asarray(guide_list,dtype='int')
    
    guide_list[exchange_list[:,0]]=exchange_list[:,1]
    #print("guide_list")
    #print(guide_list)
    l1=[-1 for i in range(N)]
    l1=np.asarray(l1,dtype='int') 
    l2=[-1 for i in range(N)]
    l2=np.asarray(l2,dtype='int')
    l1[exchange_list[:,0]]=1
    l2[exchange_list[:,1]]=1
    #print("l1: ",l1)
    #print("l2: ",l2)
    sum_l=np.add(l1,l2)
    #print("sum:  ")
    #print(sum_l)

    
    plt.clf()
    #create graph
    G=nx.DiGraph()
    exchange_list_flatten = exchange_list[:,:-1].flatten('F')
    # add nodes
    G.add_nodes_from(exchange_list_flatten)
    
    exchange_list_tuple=[tuple(e) for e in exchange_list[:,:-1] ]
    
    #add edges
    exchange_list_edges=list(zip(exchange_list[:,0], exchange_list[:,1])) 
    G.add_edges_from(exchange_list_tuple)

    #check number of nodes in each component
    node_subset=[i for i in range(N)]
    undirected_G=G.to_undirected()
    components= [list(cc) for cc in nx.connected_components(undirected_G)]
    #print("components")
    #print(components)
    
    edited_exchange_list=exchange_list.copy()
    starts=[]
    mirrors_component=[]
    same_nodes=[]
    for component in components:
        if(len(component)>2):
            start=-1
            tail=-1
            #print(component)
            for c in component:
                if len(G.in_edges(c))==0:
                    start=c
                    starts.append(start)
                if len(G.out_edges(c))==0:
                    tail=c
            if start!= -1 and tail!= -1:
                edited_exchange_list=edit_exchange_list_func(start,tail,edited_exchange_list)        
                #print("add edge between: ",start,"   ",tail)
                G.add_edge(start,tail)
        if(len(component)==1):
            same_nodes.append(component[0])
    #print("updated exchange list")
    #print(edited_exchange_list)
     #plotting the grph 
    """
    pos = nx.spring_layout(G)
    # Draw the graph according to node positions
    nx.draw(G, pos, with_labels=True)

    # Create edge labels
    labels = nx.get_edge_attributes(G,'weight')

    # Draw edge labels according to node positions
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
   """
    #plt.show()
    
    #print("same:            ",same_nodes)
    #print("starts:     ",starts)
    #print("mirrors:        \n",mirrors_component)
    for i,j in zip(sum_l,range(len(sum_l))):            
        if i == 2:
            index=np.where(exchange_list[:,0] == j)
            #print("to check:    ",edited_exchange_list[index][0][:-1])
            check = any(item in starts for item in edited_exchange_list[index][0][:-1])
            #print("check : ",check)
            same = np.all(edited_exchange_list[index][0][:-1]==edited_exchange_list[index][0][0])
            #print("same:   ",same)
            #mirror= check_in_mirrors_component(mirrors_component,edited_exchange_list[index][0])
            #print("mirror:    ",mirror)
            if (not check and not same):
                edited_exchange_list[index[0],2]=0
    
    #print(edited_exchange_list)
    for i,j in zip(edited_exchange_list[:,2],range(len(edited_exchange_list[:,2]))):
        #print(i,"    ",j)
        if i == 1:
            guide_list[edited_exchange_list[j,0]]=edited_exchange_list[j,1]
        else:
            guide_list[edited_exchange_list[j,0]] = -1
    #print("after guide_list")
    #print(guide_list)
    
    
    #setp 8
    for i,j in zip(guide_list,range(len(guide_list))):
        
        if i != -1:
            index=offspring1.index(j)
            offspring1[index]= i

    
    
    #steps 9 && 10
    f=np.zeros((N),dtype='int')
    f[offspring1]=p1
    offspring2[:]=f[p2]
    G.clear()
    for i in range(len(offspring1)-1):
        self.visited_nodes.append(offspring1[i])
        self.total_cost += self.graph.distances[offspring1[i]][offspring1[i+1]]
    self.visited_nodes.append(offspring1[N-1])
    self.total_cost += self.graph.distances[self.visited_nodes[-1]][self.visited_nodes[0]]
    #print("after: ",self.visited_nodes)
Ant.ipmx = ipmx


# In[12]:


def distance(city1: dict, city2: dict):
    return math.sqrt((city1['x'] - city2['x']) ** 2 + (city1['y'] - city2['y']) ** 2)


# # ACO parameteres initialization

# In[13]:


ant_count = 10
generations = MaxNFE
alpha = 1.0
beta = 10.0
rho = 0.5
q = 5
def ACO(method):
    
    cities = []
    points = []
    for i in range(N):        
        cities.append(dict(index=int(i), x=int(x_c[i]), y=int(y_c[i])))
        points.append((int(x_c[i]), int(y_c[i])))  
    #print(cities,"\n",points)
    distances = []
    number_of_cities = len(cities)
    for i in range(number_of_cities):
        row = []
        for j in range(number_of_cities):
            row.append(distance(cities[i], cities[j]))
        distances.append(row)
    #print(generations)
    acs = ACS(ant_count, generations, alpha, beta, rho, q)
    graph = Graph(distances, number_of_cities)
    cost, fitnesses = acs.solve(graph,method)
    return cost,fitnesses


# In[14]:


#number of running time
run_num=10


abc_two_opt=np.zeros(run_num)
abc_ipmx=np.zeros(run_num)
aco_two_opt=np.zeros(run_num)
aco_ipmx=np.zeros(run_num)

abc_two_opt_iter=np.zeros(MaxNFE)
abc_ipmx_iter=np.zeros(MaxNFE)
aco_two_opt_iter=np.zeros(MaxNFE)
aco_ipmx_iter=np.zeros(MaxNFE)


for k in range(run_num):
    print("# RUNNING: ",k) 
    abc_two_opt[k],abc_two_opt_iter=ABC(1)
    print("ABC using 2-opt: ",abc_two_opt[k])
    abc_ipmx[k],abc_ipmx_iter=ABC(2)
    print("ABC using IPMX: ",abc_ipmx[k])
    aco_two_opt[k],aco_two_opt_iter=ACO(1)
    print("ACO using 2-opt: ",aco_two_opt[k])
    aco_ipmx[k],aco_ipmx_iter=ACO(2)
    print("ACO using IMPX: ",aco_ipmx[k])


# In[ ]:


data = {'ABC twoOpt':np.around(abc_two_opt , decimals=3),
        'ABC IPMX':np.around(abc_ipmx , decimals=3),
        'ACO twoOpt':np.around(aco_two_opt , decimals=3),
        'ACO IPMX': np.around(aco_ipmx , decimals=3)
        }
df = pd.DataFrame (data, columns = ['ABC twoOpt','ABC IPMX','ACO twoOpt','ACO IPMX'])

min_0 = np.around(np.min(abc_two_opt), decimals=3)
min_1 = np.around(np.min(abc_ipmx), decimals=3)
min_2 = np.around(np.min(aco_two_opt) , decimals=3)
min_3 = np.around(np.min(aco_ipmx) , decimals=3)

# Pass the row elements as key value pairs to append() function 
df= df.append({'ABC twoOpt' : min_0 ,'ABC IPMX' : min_1 ,'ACO twoOpt' : min_2 ,'ACO IPMX' : min_3 }, ignore_index=True)


means_0 = np.around(np.mean(abc_two_opt), decimals=3)
means_1 = np.around(np.mean(abc_ipmx), decimals=3)
means_2 = np.around(np.mean(aco_two_opt) , decimals=3)
means_3 = np.around(np.mean(aco_ipmx) , decimals=3)

# Pass the row elements as key value pairs to append() function 
df= df.append({'ABC twoOpt' : means_0 ,'ABC IPMX' : means_1 ,'ACO twoOpt' : means_2 ,'ACO IPMX' : means_3 }, ignore_index=True)


median_0 = np.around(np.median(abc_two_opt), decimals=3)
median_1 = np.around(np.median(abc_ipmx), decimals=3)
median_2 = np.around(np.median(aco_two_opt) , decimals=3)
median_3 = np.around(np.median(aco_ipmx) , decimals=3)

# Pass the row elements as key value pairs to append() function 
df= df.append({'ABC twoOpt' : median_0 ,'ABC IPMX' : median_1 ,'ACO twoOpt' : median_2 ,'ACO IPMX' : median_3 }, ignore_index=True)


max_0 = np.around(np.max(abc_two_opt), decimals=3)
max_1 = np.around(np.max(abc_ipmx), decimals=3)
max_2 = np.around(np.max(aco_two_opt) , decimals=3)
max_3 = np.around(np.max(aco_ipmx) , decimals=3)
k=run_num+3
# Pass the row elements as key value pairs to append() function 
df= df.append({'ABC twoOpt' : max_0 ,'ABC IPMX' : max_1 ,'ACO twoOpt' : max_2 ,'ACO IPMX' : max_3 }, ignore_index=True)



std_0 = np.around(np.std(abc_two_opt), decimals=3)
std_1 = np.around(np.std(abc_ipmx), decimals=3)
std_2 = np.around(np.std(aco_two_opt) , decimals=3)
std_3 = np.around(np.std(aco_ipmx) , decimals=3)

# Pass the row elements as key value pairs to append() function 
df= df.append({'ABC twoOpt' : std_0 ,'ABC IPMX' : std_1 ,'ACO twoOpt' : std_2 ,'ACO IPMX' : std_3 }, ignore_index=True)
df= df.rename(index={run_num: 'Best',run_num+1: 'Average',run_num+2: 'Median',run_num+3: 'Worst',run_num+4: 'Standard Deviation'} )


print(tabulate(df, headers='keys', tablefmt='psql'))


# # Plotting

# In[ ]:


p1 = wilcoxon(abc_two_opt,abc_ipmx)
print(p1)
         
p_f1 = mannwhitneyu(abc_two_opt,abc_ipmx)
print(p_f1)

p2 = wilcoxon(abc_two_opt,aco_two_opt)
print(p2)
         
p_f2 = mannwhitneyu(abc_two_opt,aco_two_opt)
print(p_f2)

p3 = wilcoxon(abc_ipmx,aco_ipmx)
print(p3)
         
p_f3 = mannwhitneyu(abc_ipmx,aco_ipmx)
print(p_f3)


# In[ ]:



print("abc_ipmx_iter :",len(abc_ipmx_iter))
print("aco_two_opt_iter: ",len(aco_two_opt_iter))
print("aco_ipmx_iter: ",len(aco_ipmx_iter))
plt.plot(abc_two_opt_iter, label = "ABC with Two Opt")
plt.plot(abc_ipmx_iter, label = "ABC with IPMX")
plt.plot(aco_two_opt_iter, label = "ACO with Two Opt")
plt.plot(aco_ipmx_iter, label = "ACO with IPMX")
plt.legend()
plt.show()
print(abc_two_opt_iter)


# In[ ]:





# In[ ]:





# In[ ]:





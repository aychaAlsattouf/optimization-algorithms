#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter


# In[2]:


"""The objective is to minimize the vehicle fleet and the sum of travel time, 
   and the total demand of commodities for each route may not exceed the 
   capacity of the vehicle which serves that route.
NAME : A-n80-k10
COMMENT : (Augerat et al, Min no of trucks: 10, Best value: 1764)
TYPE : CVRP
DIMENSION : 80
EDGE_WEIGHT_TYPE : EUC_2D 
CAPACITY : 100 

source : http://neo.lcc.uma.es/vrp/vrp-instances/capacitated-vrp-instances/
"""
x = [92 ,88 ,70 ,57 ,0  ,61 ,65 ,91 ,59 ,3  ,95 ,80 ,66 ,79 ,99 ,20 ,40 ,50 ,97 ,21 ,36,
    100 ,11 ,69 ,69 ,29 ,14 ,50 ,89 ,57 ,60 ,48 ,17 ,21 ,77 ,2  ,63 ,68 ,41 ,48 ,98 ,26 ,
    69 ,40 ,65 ,14 ,32 ,14 ,96 ,82 ,23 ,63 ,87 ,56 ,15 ,10 ,7  ,31 ,36 ,50 ,49 ,39 ,76 ,
    83 ,33 ,0  ,52 ,52 ,46 ,3  ,46 ,94 ,26 ,75 ,57 ,34 ,28 ,59 ,51 ,87 ]

y = [92 ,58 ,6 ,59 ,98 ,38 ,22 ,52 ,2 ,54 ,38 ,28 ,42 ,74 ,25 ,43 , 3 ,42 ,0 ,19 ,21 ,61 ,
    85 ,35 ,22 ,35 ,9 ,33 ,17 ,44 ,25 ,42 ,93 ,50 ,18 ,4 ,83 ,6 ,95 ,54 ,73 ,38 ,76 ,1 ,
   41 ,86 ,39 ,24 ,5 ,98 ,85 ,69 ,19 ,75 ,63 ,45 ,30 ,11 ,93 ,31 ,52 ,10 ,40 ,34 ,51 ,15 ,
    82 ,82 ,6 ,26 ,80 ,30 ,76 ,92 ,51 ,21 ,80 ,66 ,16 ,11]
#demand
d = [0, 24, 22, 23, 5, 11, 23, 26, 9, 23, 9, 14, 16, 12, 2, 2, 6, 20, 26, 12, 15, 13, 26, 
    17, 7, 12, 4, 4, 20, 10, 9, 2, 9, 1, 2, 2, 12, 14, 23, 21, 13, 13, 23, 3, 6, 23, 11,
    2, 7, 13, 10, 3, 6, 13, 2, 14, 7, 21, 7, 22, 13, 22, 18, 22, 6, 2, 11, 5, 9, 9, 5, 
    12, 2, 12, 19, 6, 14, 2, 2, 24]
print("X: ",len(x),"\tY: ",len(y))


# In[3]:


#number of clients
n = len(x)

#clients
N = [i for i in range(1,n+1)]

#collection of vertices 
V = [0]+N

#arcs A type is list elements type is tuple
#A = [(i,j) for i in V for j in V if i!=j ]

#cost between two nodes it is the Euclidean distance between them
#np.hypot : method finds the Euclidean norm
#C type is dict
#C = {(i,j): np.hypot(x[i-1]-x[j-1],y[i-1]-y[j-1]) for i,j in A}

#Vehicle capacity
CAPACITY = 100

#demand of each client
q={i:d[j] for i,j in zip(range(1,n+1),range(n)) }

DEPOT = V[0] 

POPULATION_SIZE=80

NUMBER_OF_PARENTS=2

MUTATION_RATE = 0.5

NUM_CROSSOVER_METHODS=2

CROSSOVER_RATE = 0.7
#print(A)


# In[4]:


plt.plot(x[0],y[0],c='r',marker='s')
plt.scatter(x[1:],y[1:],c='b')


# In[5]:


def ind2route(individual):
    route = []
    vehicle_capacity = CAPACITY
    # Initialize a sub-route
    sub_route = []
    routes_count=0
    vehicle_load = 0
    last_customer_id = 0
    for customer_id in individual:
        # Update vehicle load
        demand =q.get(customer_id)
        #print("demand : ",demand)
        updated_vehicle_load = vehicle_load + demand
        #print("updated_vehicle_load : ",updated_vehicle_load)
    
        # Validate vehicle load and elapsed time
        if (demand <= vehicle_capacity):
            # Add to current sub-route
            sub_route.append(customer_id)
            vehicle_load = vehicle_load + demand
            #print("vehicle_load : ",vehicle_load)
            vehicle_capacity = vehicle_capacity -demand
            #print("vehicle_capacity : ",vehicle_capacity)
        else:
            # Save current sub-route
            #print("new vehicle")
            #print("prev vehicle take: ",vehicle_load)
            route.append(sub_route)
            # Initialize a new sub-route and add to it
            sub_route = [customer_id]
            vehicle_load = demand
            vehicle_capacity=CAPACITY -demand
            
        # Update last customer ID
        last_customer_id = customer_id
    if sub_route != []:
        # Save current sub-route before return if not empty
        route.append(sub_route)
    return route


# In[6]:


#UTILs functions
def get_two_index_randomly(start,size):
        c1 = random.randint(start, size-1)
        c2 = random.randint(start,size-1)
        
        while c1 == c2:
            c1 = random.randint(start, size-1)
            c2 = random.randint(start, size-1)
        if c2 < c1 :
            tmp=c1
            c1=c2
            c2=tmp
        return c1,c2

def print_population(population):
    for i,j in zip(population,range(1,POPULATION_SIZE+1)) :
        print("len chromosome :",len(i),"   #:",j,"    ",i)
    print("\n")

def check_repeatation(population):
    
    new_population=[]
    original_chro=[i for i in range(1,len(N)+1)]
    
    #print("original_chro : ",len(original_chro))
    
    for chromosome,i in zip(population,range(len(population))):
        #remove repeated elements
        #print("\nnewi: ",i)
        #print("result of crossover"chromosome)
        #chro=np.zeros(len(N))
        #new_chromosome= [i for i in original_chro if i not in new_chromosome ]
        indexes = np.unique(chromosome, return_index=True)[1]
        sorted_indexes=np.sort(indexes)
        insert_indexes=[]
        #print("len(sorted_indexes): ",len(sorted_indexes))
        
        insert_indexes=[i for i in range(len(chromosome)) if i not in sorted_indexes]
        new_chromosome = [chromosome[index] for index in sorted_indexes]
        #print("chromosome :  ",len(chromosome),"\tnew_chromosome: ",len(new_chromosome))
        not_exist=[]
        if len(chromosome) != len(new_chromosome):
            #print("len(chromosome) != len(new_chromosome)")
            not_exist = [i for i in original_chro if i not in new_chromosome ]
            
            #print("not_exist: ",not_exist, "\tlen: ",len(not_exist))
            for i , j in zip(range(len(not_exist)),insert_indexes) :
                #print("new_chromosome :",len(new_chromosome))
                #print("insert : ",not_exist[i],"\t at: ",j)
                new_chromosome=np.insert(new_chromosome,j,not_exist[i])
            
            #print("new chromosome:\tlen: ",len(new_chromosome))
            new_population.append(new_chromosome)
        else:
            #print("chromosome:\tlen: ",new_chromosome)
            new_population.append(chromosome)
            
    return new_population
            
        


# In[7]:


def get_distance(cus1, cus2):
    # Euclideian
    dist = 0
    cus2-=1
    cus1-=1
    #xd = x[cus1] - x[cus2]
    #yd = y[cus1] - y[cus2]
   # dist = np.sqrt( (xd * xd) + (yd * yd))
    dist = np.hypot (x[cus1]-x[cus2] , y[cus1]-y[cus2] )
    return dist

def get_fitness(li):
    
    num_custo = len(li)
    #print( num_custo)
    fitness = 0

    for i in range(num_custo-1):
        #print(li[i],"      ",li[i+1])
        fitness += get_distance(li[i], li[i+1])

    #print("DEPOT : ",DEPOT,"     get_distance(DEPOT, ",li[0],") : ",get_distance(DEPOT, li[0]))
    fitness += get_distance(DEPOT, li[0])
    fitness += get_distance(li[-1], DEPOT)
    
    # chk for valid capacity
    
    curr_demand = 0
    for i in range(num_custo):
        
        curr_demand += q.get(li[i])
        #print(q.get(li[i]))

    return fitness,curr_demand

def chromosome_fitness(chromosome):
    total_fitness=0
    vehicles_load=[]
    routes=ind2route(chromosome)
    for route in routes:
        fitness, demand =get_fitness(route)
        total_fitness+=fitness
        vehicles_load.append(demand)
    return total_fitness
def population_fitness(population):
    #print("population_fitness")
    fitnesses=[]
    for chromosome in population:
       # print(chromosome_fitness(chromosome))
        fitnesses.append(chromosome_fitness(chromosome))
    return fitnesses
#,vehicles_load , routes


def chromosome_fitness_details(chromosome):
    total_fitness=0
    vehicles_load=[]
    routes=ind2route(chromosome)
    for route in routes:
        fitness, demand =get_fitness(route)
        total_fitness+=fitness
        vehicles_load.append(demand)
    return total_fitness,vehicles_load,routes
def population_fitness_details(population):
    fitnesses=[]
    vehicles_load=[]
    routes=[]
    for chromosome in population:
       # print(chromosome_fitness(chromosome))
        f,l,r=chromosome_fitness_details(chromosome)
        fitnesses.append(f)
        vehicles_load.append(l)
        routes.append(r)
    return fitnesses,vehicles_load , routes

    


# In[8]:


def initialize_population():
    population=[]
    for i in range(POPULATION_SIZE):
        population.append(np.random.permutation(N))
    return population
        
#selection
# Stochastic Universal Sampling
def select_parents_stochastic(fitnesses): 
    
    sorted_fitnesses=np.argsort(fitnesses)
    #print(sorted_fitnesses)
    
    total_fitness =0
    for i in range(POPULATION_SIZE):
        total_fitness = total_fitness+ fitnesses[i] 
    #print(total_fitness)
    
    point_distance = total_fitness / float(NUMBER_OF_PARENTS)
    #print(point_distance)
    
    start_point = random.uniform(0, point_distance)
    #print(start_point)
    
    points = [start_point + i * point_distance for i in range(NUMBER_OF_PARENTS)]
    #print(points)

    Keep = []
    i = 0
    for P in points:
        u = random.random() * total_fitness
        #print("u  : ",u)
        sum_ = 0
        for sorted_fitness in sorted_fitnesses:
            sum_ += fitnesses[sorted_fitness]
            if sum_ > u:
                Keep.append(sorted_fitness)
                break
    return Keep

def select_the_best(fitnesses): 
    
    sorted_fitnesses=np.argsort(fitnesses)
    return sorted_fitnesses[:NUMBER_OF_PARENTS]

def selection(fitnesses):
    rand_method=random.uniform(0,1)
    if rand_method >=0.1:
        return select_the_best(fitnesses)
    else:
        return select_parents_stochastic(fitnesses)
        

def crossover_one_point(l , q ,N):

    #|print(q)
    if random.random() < CROSSOVER_RATE:

        l = list(l) 
        q = list(q) 
        # generating the random number to perform crossover 
        k = random.randint(1, N-1) 
       # print("Crossover point :", k," at the first chromosome is: ",l[k]) 

        # interchanging the genes 
        for i in range(k,N): 
            l[i], q[i] = q[i], l[i] 
        #print(l) 
        #print(q, "\n\n") 
        
        return l, q
    else:
        return 0,0

    
def crossover_two_point(l,q ,N):

    #print(crossover_two_point)
    if random.random() < CROSSOVER_RATE:
        l = list(l) 
        q = list(q) 

        # generating the random number to perform crossover 
        c1,c2=get_two_index_randomly(1,N)

        #print("Crossover points :", c1," ",c2," at the first chromosome is: ",l[c1]," ",l[c2])
        # interchanging the genes 
        for i in range(c1,c2): 
            l[i], q[i] = q[i], l[i] 
        #print(l) 
        #print(q, "\n\n") 
        return l, q
    else:
        return 0,0
    

#extra function
def population_result_crossover(crossover_population,original_population):
    population=[]
   # print(crossover_population)
    for i in range(2):
        for j in range(POPULATION_SIZE):
            #print("jjjj",j,"len crossover_population:",len(crossover_population))
            if i==0:
                population.append(crossover_population[j])
            else:
                population.append(original_population[j])
    
    fitnesses = population_fitness(population)
    sorted_fitnessess = np.argsort(fitnesses)
    population_size_select=sorted_fitnessess[:POPULATION_SIZE]
    #print("hhhhhhhhhhhhhhh",len(population_size_select))
    final_population=[population[index] for index in population_size_select]
    return final_population
    
def parents_crossover(parents_indeces, population):
    
    next_generation=[population[i] for i in parents_indeces]
    #next_generation=[]
    #generate new generation includes the crossover results and their parents
    p=[i for i in range(NUMBER_OF_PARENTS)]
    #print(p)
    i=0;
    #-NUMBER_OF_PARENTS
    while i <= POPULATION_SIZE-NUMBER_OF_PARENTS-2:
        
        random.shuffle(p)
        
        a , b = get_two_index_randomly(0,len(p))
        """
        print("first parents_index: ",parents_indeces[p[a]],"\tsecond parent index : ",parents_indeces[p[b]])
        print("population[parents_indeces[p[a]] : ",population[parents_indeces[p[a]]],"\n")
        print("population[parents_indeces[p[b]] : ",population[parents_indeces[p[b]]],"\n")
        """
        crossover_type=random.randint(0,NUM_CROSSOVER_METHODS-1)              
        if crossover_type==0:  
            c , d = crossover_one_point(population[parents_indeces[p[a]]] , population[parents_indeces[p[b]]] ,len(population[parents_indeces[p[a]]]))
            """
            print("after one point crossover a : ",c,"\n")
            print("after one point crossover b: ",d,"\n")
            """
        elif crossover_type==1:
            c , d =crossover_two_point( population[parents_indeces[p[a]]] , population[parents_indeces[p[b]]], len(population[parents_indeces[p[a]]])) 
            """
            print("after two point crossover a : ",c,"\n")
            print("after two point crossover b: ",d,"\n") 
            """
        if c and d :
            next_generation.append(c)
            next_generation.append(d)
            i+=2
    #generation=population_result_crossover(next_generation,population)
    return next_generation
            

    
def swapping_mutation(elements ,N):

    if random.random() < MUTATION_RATE :  
        
        c1=random.randint(0,N-1)
        c2=random.randint(0,N-1)
        while c1 == c2:
            c1=random.randint(0,N-1)
            c2=random.randint(0,N-1)
        #print(c1,"       ",c2)
        temp_elements=elements.copy()
        s1=elements[c1]
        s2=elements[c2]
        temp_elements[c1],temp_elements[c2]=s2,s1
        return temp_elements
    else:
        #print("kkkkkkkkkkkkkkkkkkkkk")
        return elements
        
    
def mutation(population):
    
    mutation_elements_count=random.randint(1,int((POPULATION_SIZE-1)/2))
    #print("mutation_elements_count: ",mutation_elements_count)
    
    for i in range(mutation_elements_count):
        index_of_chromosome=random.randint(0,POPULATION_SIZE-1)
        #print("index_of_chromosome: ",index_of_chromosome)
        
        population[index_of_chromosome]= swapping_mutation(population[index_of_chromosome] ,len(population[index_of_chromosome]))
        

    return population


# In[9]:


#initial_solution = np.random.permutation(N)
#will return the sum of all distances of all vehicles and vehicles load
#fitness(initial_solution)
"""population=initialize_population()
fitnesses=np.zeros(POPULATION_SIZE)
for i in range(POPULATION_SIZE):
    fitnesses[i],vehicle_load,routes=chromosome_fitness(population[i])
    """


# In[10]:


#for i in range(5):
"""print(fitnesses)
f=select_parents_stochastic(fitnesses)
print(f)
#=chromosom f[0]
print(fitnesses[f[0]])
fi,vd,rs=chromosome_fitness(population[f[0]])
print(fi)
"""


# In[15]:


def all_same(items):
     return all(x == items[0] for x in items)


def genetic_agorithm():
    population = initialize_population()
    fitnesses = population_fitness(population)
    min_fitness = fitnesses[fitnesses.index(min(fitnesses))]
    
    #print(crossover_two_point(population[0],population[1] ,len(population[0])),"\n\n")
    MAX_ITER = 40000
    i = 0
    repeat=[]
    while min_fitness >= 1774 :
        print("generation #: ",i)
        #print_population(population)
        
        #print("fitnesses")
        fitnesses = population_fitness( population)
        min_fitness = fitnesses[fitnesses.index(min(fitnesses))]
        print(min_fitness)
        repeat.append(min_fitness)
       # print(repeat)
        if i > 2000 and all_same( repeat[-1000:]):
            print("********************stack************************")
            parents = select_parents_stochastic(fitnesses)
        else:
            parents =   select_the_best(fitnesses)
            
            
        #print("parents")
        #parents = select_parents_stochastic(fitnesses)
       # parents =   selection(fitnesses)
        
        #print("parents indecs: ")
        next_generation_crossover=parents_crossover(parents,population)
        
        #print("next_generation_crossover")
        next_generation_check_repeatiation=check_repeatation(next_generation_crossover)
        
        #print("next_generation_check_repeatiation")
        next_generation_mutaion=  mutation(next_generation_check_repeatiation)
        
        #print("next_generation_mutaion")
        population=next_generation_mutaion.copy()
        i+=1
        #print_population(population)
        #print("\n",population[parents[a]])
        #print(parents[a],a,b)
    return population
    


# In[16]:


a=genetic_agorithm()


# In[13]:


fi,load,route=population_fitness_details(a)

min_index=fi.index(min(fi))

print(fi[min_index],"\tLoad: ",load[min_index],"\tvechile #:",len(route[min_index]))


# In[14]:


#for i in range(20):
 print("fitness: ",fi[min_index],"\n#of vechiles: ",len(route[min_index]),"\nload: ",load[min_index])


# In[ ]:





# In[ ]:


random.randint(0,NUM_CROSSOVER_METHODS)


# In[ ]:


a=[1,3,5,2,3,4]
#print(len(set(a)))
b= [k for k,v in Counter(a).items() if v==1]
print(b)
indexes = np.unique(a, return_index=True)[1]
print(indexes)
sorted_indexes=np.sort(indexes)
print(sorted_indexes)
insert_indexes=[]
i=0        
while i < len(sorted_indexes):
    for j in range (len(a)):
        if sorted_indexes[i] != j:
            insert_indexes.append(j)
        else:
            i+=1

print(insert_indexes)
b=[a[index] for index in np.sort(indexes)]
print(b,len(a),"\n")


# In[ ]:


b=set(a)
print(b)
repeated=[]
for i in range(len(a)):
    print(i)
    if(a[i] in b):
        print(a[i])
        repeated.append(a[i])


# In[ ]:


a=[1,23,4]
a=np.insert(a,0,5)


# In[ ]:


a=[[1,2,3,6,5,6,7,8],[4,2,1,4,5,6,7,8],[6,3,1,4,5,3,7,8]]
b=check_repeatation(a)


# In[ ]:


b


# In[ ]:


c=[a[i] for i in range(3)]


# In[ ]:


c


# In[104]:


def all_same(items):
     return all(x == items[0] for x in items)

property_list = [1.66, 3, 1.66,1.66,1.66,1.66]

if [all(x == property_list[0] for x in property_list)]:
    print("llllllll")


# In[110]:


print(property_list[-5:])


# In[ ]:





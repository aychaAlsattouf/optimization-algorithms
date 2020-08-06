#!/usr/bin/env python
# coding: utf-8

# In[11]:


#depends on http://www.cs.ubc.ca/labs/beta/Courses/CPSC532D-03/Resources/StuHoo99.pdf

# result will depends on how many iterations required to reach to the acceptable value
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import itertools


# In[12]:


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


#N = len(x_c)

MAX_ITER =30000

#number of clients
n = len(x)

#clients
N = [i for i in range(1,n)]

#collection of vertices 
V = [0]+N


#Vehicle capacity
CAPACITY = 100

#demand of each client
#q={i:d[j] for i,j in zip(range(1,n+1),range(n)) }

DEPOT = V[0] 


# In[13]:


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


# In[14]:


def ind2route(individual):
    #print("individual : ",individual)
    route = []
    vehicle_capacity = CAPACITY
    vehicle_load = 0
    # Initialize a sub-route
    sub_route = []
    routes_count=0
   
    last_customer_id = 0
    for customer_id in individual:
        # Update vehicle load
        demand = d[customer_id]
    
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
            #print("vehicle_load : ",vehicle_load)
            #print(sub_route)
            route.append(sub_route)
            # Initialize a new sub-route and add to it
            sub_route = [customer_id]
            vehicle_load = demand
            vehicle_capacity = CAPACITY -demand
            
        # Update last customer ID
        last_customer_id = customer_id
    if sub_route != []:
        # Save current sub-route before return if not empty
        #print("vehicle_load : ",vehicle_load)
        route.append(sub_route)
    return route


def get_distance(cus1, cus2):
    # Euclideian
    dist = 0
    #cus2-=1
    #cus1-=1
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
        #print(get_distance(li[i], li[i+1]))

    fitness += get_distance(DEPOT, li[0])
    fitness += get_distance(li[-1], DEPOT)
    #print(fitness)
    # chk for valid capacity
    
    curr_demand = 0
    for i in range(num_custo):
        
        curr_demand += d[li[i]]
        #print(q.get(li[i]))

    return fitness,curr_demand

def fitness(chromosome):
    total_fitness=0
    vehicles_load=[]
    routes=ind2route(chromosome)
    for route in routes:
    
        fitness, demand =get_fitness(route)
        total_fitness+=fitness
        vehicles_load.append(demand)
    return total_fitness


def fitness_details(chromosome):
    total_fitness=0
    vehicles_load=[]
    routes=ind2route(chromosome)
    for route in routes:
        fitness, demand =get_fitness(route)
        total_fitness+=fitness
        vehicles_load.append(demand)
    return total_fitness,vehicles_load,routes


# In[15]:


def closestpoint(point, route):
    dmin = float("inf")
    for p in route:
        xd_0 = x[point] - x[p]
        yd_0 = y[point] - y[p]
        #d = np.sqrt( (xd_0 * xd_0) + (yd_0 * yd_0))
        d = get_distance(point,p)
        #np.hypot (x[point]-x[p] , y[point]-y[p] )
         #math.sqrt((point[0] - p[0])**2 + (point[1] - p[1])**2)
        if d < dmin:
            dmin = d
            closest = p
    return closest
def nearestneighbor(route):
    
    path = []
    sub_path = [DEPOT]
    vehicle_capacity = CAPACITY
    total_dist=0
    vehicle_load=0
    while len(route) >= 1:
        
        closest= closestpoint(sub_path[-1], route)
        demand = d[closest]
        
        if(demand <= vehicle_capacity):
        
            path.append(closest)
            sub_path.append(closest)
            closest_index= np.where(route== closest)
            route=  np.delete(route, closest_index)
            vehicle_capacity= vehicle_capacity - demand
            
        else:
            #print(sub_path)
            closest = closestpoint(sub_path[-1], [DEPOT])
            sub_path = [DEPOT]
            vehicle_capacity = CAPACITY
            demand = 0
    if sub_path != []:
        closest = closestpoint(sub_path[-1], [DEPOT])
        
    return path


# In[16]:


def initial_solution():
    route = np.random.permutation(N)
    initial_route = nearestneighbor(route)
    return initial_route


# In[17]:


def reverse_segment_if_better(tour, i, j, k):
    """If reversing tour[i:j] would make the tour shorter, then do it."""
    # Given tour [...A-B...C-D...E-F...]
    A, B, C, D, E, F = tour[i-1], tour[i], tour[j-1], tour[j], tour[k-1], tour[k % len(tour)]
    d0 = get_distance(A, B) + get_distance(C, D) + get_distance(E, F)
    d1 = get_distance(A, C) + get_distance(B, D) + get_distance(E, F)
    d2 = get_distance(A, B) + get_distance(C, E) + get_distance(D, F)
    d3 = get_distance(A, D) + get_distance(E, B) + get_distance(C, F)
    d4 = get_distance(F, B) + get_distance(C, D) + get_distance(E, A)

    if d0 > d1:
        tour[i:j] = list(reversed(tour[i:j]))
        #print(d0,"    ",d1)
        return -d0 + d1
    elif d0 > d2:
        #print(d1,"    ",d2)
        tour[j:k] = list(reversed(tour[j:k]))
        return -d0 + d2
    elif d0 > d4:
        #print(d0,"    ",d4)
        tour[i:k] = list(reversed(tour[i:k]))
        return -d0 + d4
    elif d0 > d3:
        #print(d0,"    ",d3)
        tmp = np.concatenate((tour[j:k],  tour[i:j]), axis=0) 
        tour[i:k] = tmp
        return -d0 + d3
    return 0
def three_opt(tour):
    """Iterative improvement based on 3 exchange."""
    while True:
        delta = 0
        for (a, b, c) in all_segments(len(tour)):
            delta += reverse_segment_if_better(tour, a, b, c)
            #print(delta)
        if delta >= 0:
            break
    return tour

def all_segments(n: int):
    """Generate all segments combinations"""
    return ((i, j, k)
        for i in range(n)
        for j in range(i + 2, n)
        for k in range(j + 2, n + (i > 0)))

def change_tour(num1,num2,tour):
    if num1 > num2:
        temp = num1
        num1 = num2
        num2 = temp
    temp_tour=tour.copy()
    temp_tour[num1:num2+1]=list(reversed(temp_tour[num1:num2+1]))
    return temp_tour
    

def two_opt(tour):
    #select index to change
    best_tour= tour.copy()

    combinations =itertools.combinations(tour, 2)
    for combination in combinations:
        combination = list(combination)
        n1=combination[0]
        n2=combination[1]
        new_tour = change_tour(n1,n2,tour)
       # print("fitness(new_tour): ",fitness(np.array(new_tour)))
    #for k in range(100):
     #   n1,n2 =get_two_index_randomly(1,n)
        new_tour = change_tour(n1,n2,tour)
        if fitness(np.array(new_tour)) <= fitness(np.array(best_tour)):
            #print("fitness(new_tour): ",fitness(np.array(new_tour)),"          fitness(best_tour) : ",fitness(np.array(best_tour)))
            best_tour= new_tour.copy()
         #   break
            
    return best_tour


# In[18]:



def modify(tour):
    #print(tour)
    quarter= n//4
    #print(quarter)
    q1 = tour[ : quarter]
    #print(len(q1))
    q2 = tour[quarter : quarter*2]
    q3 = tour[quarter*2 : quarter*3]
    q4 = tour[quarter*3 : ]
    #print("q1: ",q1,"\nq2: ",q2,"\nq3: ",q3,"\nq4: ",q4)
    new_tour = tour.copy()
    new_tour[ : len(q4)] = q4
    new_tour[len(q4) : len(q4)+quarter] = q3
    new_tour[len(q4)+quarter : len(q4)+quarter*2] = q2
    new_tour[len(q4)+quarter*2 : ] = q1
    
    #random.shuffle(tour)
    #print("new tour: ",tour)
    return new_tour

def all_same(items):
    
    return all(x == items[0] for x in items)

def local_search(route):
    
    opt= random.randint(2,3)

    if opt ==2:
       # print("two")
        new_tour = two_opt(np.array(route))
    else:
        new_tour = three_opt(np.array(route))
    return new_tour
    
def better_equal(solution,solution2 , repeated):
   # print(fitness(solution),"        ",fitness(solution2))
    worse_prob = 0.09
    rand_prob = np.random.uniform(0,1,1)
    if fitness(solution2) < fitness(solution):
        #print("new")
        return solution2
    elif repeated:
       # print("prob")
        return solution2
    else:
        #print("solution")
        return solution
    


# In[19]:



#print(s)
#print(fitness(s))
repeat=[]
fitnesses=[]
iterations = []
MAX_ITER = 5000
for i in range (25):
    j = 0
    tour = initial_solution()
    s = local_search(tour)
    while fitness(s) > 2000:
        #print("ITER # ",i,"     ",fitness(s))
        fitnesses.append(fitness(s))
        #if fitness(s) < 1790:
         #   break
        repeat.append(fitness(s))
        
        repeated_value=all_same( repeat[-10:])
        new_s = s.copy()
        s1 = modify(new_s)
        s2 = local_search(s1)
        s = better_equal( s , s2 , repeated_value)
        #print(fitness(s))
        j += 1
    print("iterations = ",j)
    print(fitness(s),"\n")
    iterations.append(j)
        #print(s)
        #if fitness(s) < 1800:
        #    break


# In[20]:


np.mean(iterations)


# In[ ]:



  
 # print(route)
  sub_path=[DEPOT]
  
  path=[]
  sum = 0
  sum_sub=0
  vehicle_load = 0
  vehicle_capacity = CAPACITY
  while len(route) >= 1:
      
      #print(sub_path)
      closest, dist = closestpoint(sub_path[-1], route)
      #print(closest)
      demand = q.get(closest)
     
      #print("updated_vehicle_load : ",updated_vehicle_load)
      #print("sub_path: ",sub_path)
      #print("updated_vehicle_load:  " ,updated_vehicle_load)
     # print("closest: ",closest,"   vehicle_capacity: ",vehicle_capacity,"     demand: ",demand)
      # Validate vehicle load and elapsed time
      if (demand <= vehicle_capacity):
          
          # Add to current sub-route
          path.append(closest)
          sub_path.append(closest)
          closest_index= np.where(route== closest)
          route=  np.delete(route, closest_index)
          sum += dist
          sum_sub+=dist
          vehicle_load = vehicle_load + demand
          vehicle_capacity = vehicle_capacity - demand
          print("demand : ",demand, "         vehicle_capacity:",vehicle_capacity)
          
          
      else:
          print(sub_path)
          #print("vehicle_load : ",vehicle_load)
          closest, dist = closestpoint(sub_path[-1], [DEPOT])
          sum += dist
          sum_sub+=dist
          #print("sub_sum: ",sum_sub)
          sum_sub=0
          sub_path = [DEPOT]
          closest, dist = closestpoint(sub_path[-1], route)
      #print(closest)
          demand = q.get(closest)
          vehicle_load = demand
          vehicle_capacity = CAPACITY - demand 
  print(sub_path)
  print("vehicle_load : ",vehicle_load)


# In[ ]:





# In[ ]:





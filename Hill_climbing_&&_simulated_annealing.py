#!/usr/bin/env python
# coding: utf-8

# In[2]:



import math
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import neighbors_operations


# In[4]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import neighbors_operations


# In[2]:


from pylab import rcParams
rcParams['figure.figsize'] = 15,10


# Define Nodes Coordinate 

# In[3]:


x=[565.0 , 25.0 ,345.0 ,945.0 , 845.0 , 880.0 , 25.0 ,  525.0 , 580.0 , 650.0 , 1605.0 , 1220.0 ,  1465.0 ,1530.0 , 845.0 ,
   725.0 , 145.0 , 415.0 , 510.0 , 560.0 , 300.0 , 520.0 , 480.0 , 835.0 , 975.0 , 1215.0 , 1320.0 ,  1250.0 ,  660.0 , 
   410.0 , 420.0 , 575.0 , 1150.0 ,700.0 , 685.0 , 685.0 , 770.0 , 795.0 , 720.0 , 760.0 , 475.0 , 95.0 ,  875.0 , 700.0 ,
   555.0 , 830.0 , 1170.0 ,  830.0 , 605.0 , 595.0 , 1340.0 , 1740.0 ]

y=[575.0 , 185.0 , 750.0 , 685.0 , 655.0 , 660.0 , 230.0 , 1000.0 , 1175.0 , 1130.0 , 620.0 ,
   580.0 , 200.0 , 5.0 , 680.0 , 370.0 , 665.0 , 635.0 , 875.0 ,365.0 , 465.0 , 585.0 , 415.0 , 625.0 , 580.0 ,
   245.0 , 315.0 , 400.0 , 180.0 , 250.0 , 555.0 , 665.0 , 1160.0 , 580.0 , 595.0 , 610.0 , 610.0 , 645.0 , 635.0 , 
   650.0 , 960.0 , 260.0 , 920.0 , 500.0 , 815.0 , 485.0 , 65.0 , 610.0 , 625.0 , 360.0 , 725.0 , 245.0 ]

print("Data Size\nX: ",len(x),"\tY: ",len(y))


# ## EUC distance objective function

# In[4]:


def TSP_fun(p , N):
    
    xd_0=x[p[N-1]] - x[p[0]]
    yd_0=y[p[N-1]] - y[p[0]]
    distance = np.sqrt( (xd_0 * xd_0) + (yd_0 * yd_0))
    for i in range(N - 1):
        xd = x[p[i]] - x[p[i+1]]
        yd = y[p[i]] - y[p[i+1]]
        
        dxy = np.sqrt( (xd * xd) + (yd * yd))
        distance = distance + dxy
        
    return distance               




# # Hill Climbing 

# In[8]:


"""
p = initial solution
hill_climbing_ITER = # of iterations
neighbor_Op = neighbors Operation
    1.sawpping
    2.2-opt
    3.insertion
"""
def hill_climbing(cities_count ,p ,hill_climbing_ITER ,neighbor_Op):
    p_new = p.copy()
    fitness = TSP_fun(p_new,cities_count)

    best_p = p.copy()
    best_fitness = fitness

    hill_all_fitness = np.zeros(hill_climbing_ITER)

    iteration = 0
    hill_all_fitness[0] = best_fitness
    while iteration < hill_climbing_ITER:
        if(neighbor_Op==1):
            p_new  = neighbors_operations.swapping_fun( p , cities_count) 
        elif(neighbor_Op==2):
            p_new  = neighbors_operations.opt_2_fun( p , cities_count)
        elif(neighbor_Op==3):
            p_new  = neighbors_operations.insertion_fun( p , cities_count)

        fitness_new = TSP_fun(p_new,cities_count)

        if fitness_new < best_fitness:
            best_fitness = fitness_new
            best_p = p_new.copy()
            p = p_new.copy()
        hill_all_fitness[iteration] = best_fitness
        iteration = iteration + 1
    
    #print(p_best)
    op=""
    if(neighbor_Op==1):
        op="Swapping"
    elif(neighbor_Op==2):
        op="2-opt"
    elif(neighbor_Op==3):
        op="Insertion"
    
    print("Hill Climbing (",op,") : ",hill_all_fitness[-1])
    return hill_all_fitness[-1],hill_all_fitness , best_p


# # Simulated Annealing

# In[9]:


"""
p = initial solution
hill_climbing_ITER = size of fitness array
T_initial = initial temp
simulated_annealing_ITER =# of simulated annealing ITER
neighbor_Op = neighbors Operation
    1.sawpping
    2.2-opt
    3.insertion
"""
def simulated_annealing(cities_count ,p ,hill_climbing_ITER , T_initial ,a,simulated_annealing_ITER,neighbor_Op):

    p_copy=p.copy()
    p_best=p.copy()

    fitness=TSP_fun(p,cities_count)

    simulated_all_fitnesses=np.zeros(hill_climbing_ITER )
    iterations=0
    simulated_all_fitnesses[0]=fitness
    best_fitness=fitness


    for i in range (simulated_annealing_ITER):
        T = T_initial
        while T > 0.001 :
                p_copy = p.copy()
                if(neighbor_Op==1):
                    p_copy  = neighbors_operations.swapping_fun( p , cities_count) 
                elif(neighbor_Op==2):
                    p_copy  = neighbors_operations.opt_2_fun( p , cities_count)
                elif(neighbor_Op==3):
                    p_copy  = neighbors_operations.insertion_fun( p , cities_count)
                fitness_new = TSP_fun(p_copy , cities_count)

                if fitness_new < fitness :
                    fitness = fitness_new
                    p = p_copy.copy()

                    if fitness_new < best_fitness :
                        best_fitness = fitness_new
                        p_best = p.copy()

                elif np.random.uniform() < np.exp(-(fitness_new-fitness)/T):
                    fitness = fitness_new
                    p = p_copy.copy()


                simulated_all_fitnesses[iterations] = best_fitness
                iterations = iterations + 1

                T = T * a
    #print(p_best)
    op=""
    if(neighbor_Op==1):
        op="Swapping"
    elif(neighbor_Op==2):
        op="2-opt"
    elif(neighbor_Op==3):
        op="Insertion"
    print("Simulated Annealing (",op,"): ",simulated_all_fitnesses[iterations-1])
    
    return simulated_all_fitnesses[iterations-1],simulated_all_fitnesses , p_best


# In[10]:


cities_count=len(x)

#number of iteration for hill climbing
h_c_ITER = 183300

#number of iteration for simulated annealing
s_a_ITER = 100
Temp = 100000
a = 0.99


# loop for creating comparison table

# In[11]:


# number of running times
n=10

#best path for each method
best_hill_swap_path = 0.0
best_hill_2_opt_path = 0.0
best_hill_insertion_path = 0.0
best_simulated_swap_path = 0.0
best_simulated_2_opt_path = 0.0  
best_hill_insertion_path = 0.0

#fitness in all iteration for all methods
simulated_swap_ITER = np.zeros(h_c_ITER)
simulated_insertion_ITER = np.zeros(h_c_ITER)
simulated_2_opt_ITER= np.zeros(h_c_ITER)
hill_swap_ITER = np.zeros(h_c_ITER)
hill_2_opt_ITER = np.zeros(h_c_ITER)
hill_insertion_ITER = np.zeros(h_c_ITER)

#value of last iteration for each running time
simulated_swap =np.zeros(n)
simulated_insertion = np.zeros(n)
simulated_2_opt = np.zeros(n)
hill_swap = np.zeros(n)
hill_2_opt = np.zeros(n)
hill_insertion = np.zeros(n)

for i in range(n):
    print("# Iter : ",i)
    #new initial solution for each running
    p_initial = np.random.permutation(cities_count)
    hill_swap[i],hill_swap_ITER ,best_hill_swap_path = hill_climbing(cities_count,p_initial,h_c_ITER,1)
    hill_2_opt[i], hill_2_opt_ITER , best_hill_2_opt_path= hill_climbing(cities_count,p_initial,h_c_ITER,2)
    hill_insertion[i], hill_insertion_ITER ,best_hill_insertion_path = hill_climbing(cities_count,p_initial,h_c_ITER,3)
    
    simulated_swap[i] , simulated_swap_ITER , best_simulated_swap_path = simulated_annealing(cities_count , p_initial , h_c_ITER , Temp , a ,s_a_ITER,1)
    simulated_2_opt[i] , simulated_2_opt_ITER , best_simulated_1_opt_path = simulated_annealing(cities_count ,p_initial , h_c_ITER , Temp , a ,s_a_ITER,2)
    simulated_insertion[i] , simulated_insertion_ITER , best_simulated_insertion_path = simulated_annealing(cities_count , p_initial ,h_c_ITER , Temp , a , s_a_ITER,3)


# In[12]:


data = {'HC swap':np.around(hill_swap , decimals=3),
        'HC 2_opt':np.around(hill_2_opt , decimals=3),
        'HC insert':np.around(hill_insertion , decimals=3),
        'SA swap': np.around(simulated_swap , decimals=3),
        'SA 2_opt': np.around(simulated_2_opt , decimals=3),
        'SA insert':  np.around(simulated_insertion , decimals=3),
        }
df = pd.DataFrame (data, columns = ['HC swap','HC 2_opt','HC insert','SA swap','SA 2_opt','SA insert'])


means_0 = np.around(np.mean(hill_swap), decimals=3)
means_1 = np.around(np.mean(hill_2_opt), decimals=3)
means_2 = np.around(np.mean(hill_insertion) , decimals=3)
means_3 = np.around(np.mean(simulated_swap) , decimals=3)
means_4 = np.around(np.mean(simulated_2_opt) , decimals=3)
means_5 = np.around(np.mean(simulated_insertion) , decimals=3)

# Pass the row elements as key value pairs to append() function 
df= df.append({'HC swap' : means_0 ,'HC 2_opt' : means_1 ,'HC insert' : means_2 ,'SA swap' : means_3 ,'SA 2_opt' :  means_4 ,'SA insert' :  means_5} , ignore_index=True)
df= df.rename(index={n: 'Average'} )

print(tabulate(df, headers='keys', tablefmt='psql'))


# # Drawing the results of all methods

# In[13]:


fig,axs = plt.subplots(2,3)
axs[0,0].set_title('Hill Climbing for TSP (Swapping)')
axs[0,0].plot(hill_swap_ITER , color="red")
axs[0,1].set_title('Hill Climbing for TSP (2_opt)')
axs[0,1].plot(hill_2_opt_ITER ,color="green")
axs[0,2].set_title('Hill Climbing for TSP (Insertion)')
axs[0,2].plot(hill_insertion_ITER , color="yellow")

axs[1,0].set_title('Simulated Annealing for TSP (Swapping)')
axs[1,0].plot(simulated_swap_ITER,color="pink")
axs[1,1].set_title('Simulated Annealing for TSP (2_opt)')
axs[1,1].plot(simulated_2_opt_ITER , color="blue")
axs[1,2].set_title('Simulated Annealing for TSP (Insertion)')
axs[1,2].plot(simulated_insertion_ITER , color="purple")
for i in range(2):
    for j in range (3):
        axs[i,j].grid()
plt.show()


# ## Drawing average of all methods

# In[14]:


averages=[means_0,means_1,means_2,means_3,means_4,means_5]
names=['HC swap','HC 2_opt','HC insert', 'SA swap','SA 2_opt','SA insert']
plt.bar(names, averages ,  color=['red', 'green', 'yellow', 'pink', 'blue','purple'])
plt.title("compare algorithms averages")
plt.ylabel("Averages")
plt.xlabel("Algorithms")
rcParams['figure.figsize'] = (5.0, 8.0)
plt.show()


# In[ ]:





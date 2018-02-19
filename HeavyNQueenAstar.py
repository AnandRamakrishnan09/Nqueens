import numpy as np
import time
from queue import PriorityQueue
import random
import math
global pq
import copy
pq=PriorityQueue()
global pq_list
pq_list=[]


class Node:
    def __init__(self,N=0):
    
        self.state=np.zeros((N,N))
        self.g_x=0
        self.h_x=0
        self.cost_so_far=0
        self.parent= None

def is_goal(state):
    #checking same row
    #checking same column
    #checking diagonal1
    #checking diagonal
    #checking row on left side
    x=np.where(state == 1)
    N=len(state)
    for i in range (len(x[0])):
        #print (x[0][i],x[1][i])
        row=x[0][i]
        col=x[1][i]
        
        count=0
        for i in range(N):
            if(state[row][i]==1):
                count=count+1
        
        if count>1:
            return 0
        
        count=0
        for i in range(N):
            if(state[i][col]==1):
                count=count+1
        
        if count>1:
            return 0
        
        count=0
       
        #Primary Diagonal 
        row1=row-1
        col1=col-1
        
        while row1>=0 and row1<N and col1>=0 and col1<N:
           
            if(state[row1][col1]==1):
                return 0
            row1=row1-1
            col1=col1-1
        
        row1=row+1
        col1=col+1
        
        while row1>=0 and row1<N and col1>=0 and col1<N:
            
            if(state[row1][col1]==1):
                return 0
            row1=row1+1
            col1=col1+1
            
        row1=row-1
        col1=col+1
        
        #Secondary diagonal
        while row1>=0 and row1<N and col1>=0 and col1<N:
            
            if(state[row1][col1]==1):
                return 0
            row1=row1-1
            col1=col1+1
            
        row1=row+1
        col1=col-1
        
        while row1>=0 and row1<N and col1>=0 and col1<N:
            
            if(state[row1][col1]==1):
                return 0
            row1=row1+1
            col1=col1-1
            
    return 1

def random_state(N):
    
    board=np.zeros((N,N))
    print("Creating Random First State for N =",N)
    for x in range(N):
        row=random.randrange(0,N)
        #print(row,x)
        board[row][x]=1
    #board=[[0,1,0,0],[1,0,0,1],[0,0,0,0],[0,0,1,0]]
    #board=[[1,1,0,0],[0,0,0,1],[0,0,0,0],[0,0,1,0]]
    #board=[[1,0,0,0],[0,1,0,1],[0,0,0,0],[0,0,1,0]]
    #board=[[1,1,1,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]]
    #board=[[0,1,1,0],[1,0,0,0],[0,0,0,1],[0,0,0,0]]
    #board=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    #board=[[0,1,0,0],[0,0,0,1],[0,0,1,0],[1,0,0,0]]
    #board=[[0,0,0,1],[0,1,0,0],[0,0,0,0],[1,0,1,0]]
    #board=[[1,0,0,0],[0,0,0,0],[0,1,1,0],[0,0,0,1]]
    #board=[[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,1,1,1]]
    #not working:
    #board=[[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]]
    #board=np.array(board)
    return board

def cal_heuristic(state):
    N=len(state)
    x=np.where(state == 1)
    attack=[]
    count=0
    for i in range (len(x[0])):
        #print (x[0][i],x[1][i])
        row=x[0][i]
        col=x[1][i]
        
        #same row
        
        for i in range(N):
            if(state[row][i]==1 and i!=col):
                
                count=count+1
                #print("Same Row",count)
                attack.append([[row,col],[row,i]])
        
        #same column
        for i in range(N):
            if(state[i][col]==1 and i!=row):
               # print("Same Column",count)
                count=count+1
                attack.append([[row,col],[row,i]])
        
        

        
        #Primary diagonal - upper left
        row1=row-1
        col1=col-1
        
        while row1>=0 and row1<N and col1>=0 and col1<N:
            
            if(state[row1][col1]==1):
                count=count+1
                #print("Same primary diagonal1",count)
                attack.append([[row,col],[row1,col1]])
            row1=row1-1
            col1=col1-1
        
        #lower right
        row1=row+1
        col1=col+1
        
        while row1>=0 and row1<N and col1>=0 and col1<N:
            
            if(state[row1][col1]==1):
                count=count+1
                #print("Same primary diagonal2",count)
                attack.append([[row,col],[row1,col1]])
        
            row1=row1+1
            col1=col1+1
            
        row1=row-1
        col1=col+1
        
        #Secondary diagonal - upper right
        while row1>=0 and row1<N and col1>=0 and col1<N:
            
            if(state[row1][col1]==1):
                count=count+1
                #print("Same secondary diag",count)
                attack.append([[row,col],[row1,col1]])
        
            row1=row1-1
            col1=col1+1
            
        row1=row+1
        col1=col-1
        
        #lower left
        while row1>=0 and row1<N and col1>=0 and col1<N:
            
            if(state[row1][col1]==1):
                count=count+1
                #print("Same secondary diag",count)
                attack.append([[row,col],[row1,col1]])
        
            row1=row1+1
            col1=col1-1
    #print("Cal Heuristic",count/2)
    #print(attack)
    temp=math.floor((count/2))
    if temp==0:
        return 0
    return (10+math.floor((count/2)))


def cal_g(state1,state2):
    #print("Inside Cal_g")
    #print (state1)
    #print (state2)
    if (np.array_equal(state1,state2)==0):
        changed_state=np.absolute(state1-state2)
        ones=np.where(changed_state==1)[0]
        ones2=np.where(changed_state==1)[1]
        #print (ones)
        #print (ones2)
        diff=abs(ones[1]-ones2[1])
        #print (diff)
        return (10+(diff*diff))
    else:
        return 10

def populate(x):
    global pq
    global pq_list
    a=None
    temp=None
    state=np.copy(x.state)
    for col in range(len(state)):
        state=np.copy(x.state)
        for row in range(len(state)):
            temp=None
            
            temp=Node()
            temp.parent=x
            
            #state[col][:]=0
            for k in range(len(state)):
                state[k][col]=0
            
            #print (state)
            state[row][col]=1
            #print("Inside Populate")
            #print(state)
            temp.state=state
            #print ("Costs")
            #print (temp.h_x)
            #print (temp.g_x)
            #print (x.g_x)
            temp.h_x=cal_heuristic(temp.state)
            temp.g_x=cal_g(temp.state,x.state)
            #print (temp.g_x)
            #print (temp.h_x)
            temp.cost_so_far=temp.h_x+temp.g_x+x.g_x
            #print (temp)
            #print (temp.cost_so_far)
            #print("I am printing in populate")
            #print (temp.state)
            if np.array_equal(x.state,temp.state)==0:
                a=copy.deepcopy(temp)
                pq_list.append(a)
            #print(len(pq_list)-1)
                
                pq.put((temp.cost_so_far,len(pq_list)-1))

def print_soln(state1):
    list_soln=[]
    while(state1.parent!=None):
        list_soln.append(state1.state)
        state1=state1.parent
    print ("Branching Factor",float(len(pq_list)/len(list_soln)))
    while(len(list_soln)!=0):
        #print("Printing soln")
        a=list_soln.pop()
        print(a)

print("Enter N")
N=int(input())
time_start=time.clock()
start_state=random_state(N);
print("INITIAL START STATE")
print(start_state)
start=Node()
start.state=start_state
while(True):
    populate(start)
    next_indice=pq.get()
    #print("Indice",next_indice[1])
    next_state=pq_list[next_indice[1]]
    #print("Next STate")
    #print (next_indice[0])
    #print(next_state.state)
    if(is_goal(next_state.state)==1):
        time_end=time.clock()
        print("Solution reached")
        print_soln(next_state)
        print("Number of Nodes expanded",len(pq_list))
        print ("Effective Cost to solve the problem",next_state.cost_so_far)
        print ("Total Time Taken",time_end-time_start)
        break
    
    start=next_state
    


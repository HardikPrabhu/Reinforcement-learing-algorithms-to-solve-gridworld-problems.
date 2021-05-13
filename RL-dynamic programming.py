import numpy


#state space
states=[]
for i in range(1,7):
    for j in range(1,7):
        if (i,j)!=(5,3) and (i,j)!=(4,3):
            states.append((i,j))

 #action space
actions=["L","U","R","D"]



#how  agent should move (not under the influence of the environment)
#without any boundaries, if action=Left, state=(3,3) then output:(3,2) 

def move(action, state): 
    if action == "L":
        return state[0], state[1] - 1
    if action == "R":
        return state[0], state[1] + 1
    if action == "U":
        return state[0] - 1, state[1]
    if action == "D":
        return state[0] + 1,state[1]

#checking for states in which we are legally allowed to move

def legal(new_state):  
    if new_state[0]<1 or new_state[0]>6 or new_state[1]<1 or new_state[1]>6:
        return False #invalid transition (hitting a wall)
    if new_state==(4,3) or new_state==(5,3):
        return False #invalid transition (falling into the hole)
    return True
    
#setting up the transition probabilities
#given a state and an action returns the dictionary of probabilities of entering the new possible states
def transprob(state,action):
    a=["L","U","R","D"]
    prob={}
    prob[state]=0.1 #returning to the same state
    if legal(move(action,state)) == True:
        prob[move(action,state)]=0.8
    else:
        prob[state]=prob[state]+0.8
    #rotating clockwise
    clk=(a.index(action)+1)%4
    if legal(move(a[clk],state))==True:
        prob[move(a[clk],state)]=0.05
    else:
        prob[state]=prob[state]+0.05

    #rotating anti-clockwise
    anti_clk=(a.index(action)-1)%4
    if legal(move(a[anti_clk], state)) == True:
        prob[move(a[anti_clk], state)] = 0.05
    else:
        prob[state] = prob[state] + 0.05

    if state==(6,6): #terminal state
        prob={}
        prob[state]=1
    return prob
    
    
    
#initializing value function

value={}
for i in range(1,7):
    for j in range(1,7):
       value[(i,j)]=0


def update_value(value,state,gamma):
    a = ["L", "U", "R", "D"]
    sum=0
    for action in a:
        prob=transprob(state,action) #dictinary for new states prob
        x=0
        for new_state in prob:
            reward=0
            if new_state==(6,6):
                reward=15
            if new_state==(6,3):
                reward=-15
            x=x+prob[new_state]*(gamma*value[new_state]+reward)
        sum=sum+0.25*x

    value[state]=sum

    return
  
#Policy evaluation-------------------------------------------------------------------------------------------------------  
  #we will update the values sweeping through all the valid states, values of terminal state and the holes are zeros.
repeat=True
while repeat==True:
 delta=0
 for state in value:
    if state!=(6,6) and state!=(4,3) and state!=(5,3):
        old=value[state]
        update_value(value,state,0.9)
        delta=max(abs(value[state]-old),delta)
 if delta<0.0005:
    repeat=False

    
# creating the Q values using the value function

#after policy evaluation!!!
def Q_value(state, action,gamma):
    if state != (6, 6) and state != (4, 3) and state != (5, 3):
     prob = transprob(state, action)
     x = 0
     for new_state in prob:
        reward = 0
        if new_state == (6, 6):
            reward = 15
        if new_state == (6, 3):
            reward = -15
        x =x+prob[new_state] * (gamma * value[new_state] + reward)
     return x
    return 0




Q={}
for i in states:
    for a in actions:
       Q[i,a]=Q_value(i,a,0.9)   #0.9 is our discount factor     
    
    
#part b policy iteration---------------------------------------------------------------------------




val={}  #dictionary of state-values

#randomly initizalize it
import random

for i in range(1,7):
    for j in range(1,7):
       state=(i,j)
       if state != (6, 6) and state != (4, 3) and state != (5, 3):
                val[state]=random.random()

    val[(6,6)]=15           #as per the question


# we start with the same policy as the previous question
policy={}
for state in val:
 policy[state]=[0.25, 0.25, 0.25, 0.25]  #for selecting actions [L,U,R,D]

def update_val(val,state,gamma):
    a = ["L", "U", "R", "D"]
    sum=0
    for action in a:
        prob=transprob(state,action) #dictinary for new states prob
        x=0
        for new_state in prob:
            reward=0
            if new_state==(6,6):
                reward=15
            if new_state==(6,3):
                reward=-15
            x=x+prob[new_state]*(gamma*val[new_state]+reward)
        sum=sum+policy[state][a.index(action)]*x       #according to policy

    val[state]=sum

    return

value_11=[]
for i in range(100):
 repeat=True
 while repeat==True:
  delta=0
  for state in val:
    if state!=(6,6) and state!=(4,3) and state!=(5,3):
        old=val[state]
        update_value(val,state,0.9)
        delta=max(abs(val[state]-old),delta)
  if delta<0.0005:
    repeat=False
  #policy improvement
 stable=False
 while stable==False:
   stable=True
   for state in policy:
    pie=max(policy[state])
    pie_s=[i for i, x in enumerate(policy[state]) if x == pie]
    a=["L","U","R","D"]
    q=[]
    for action in a:
        q.append(Q_value(state,action,0.9))
    maxq=max(q)
    candidates=[i for i, x in enumerate(q) if x == maxq]
    policy[state]=[0,0,0,0]
    for i in candidates:
        policy[state][i]=1/len(candidates)
    if candidates!=pie_s:
        stable=False
 value_11.append(val[(1,1)])


#optimal action
op_action={}
a=["L","U","R","D"]
for i in policy:
   if i!=(6,6):
    x=policy[i]
    for j in range(len(x)):
      if x[j]>0:
          op_action[i]=a[j]

#printing optimal policy in form of  a matrix:
import numpy as np
mat=np.zeros((6,6),dtype=object)
for j in op_action:
    mat[j[0]-1][j[1]-1]=op_action[j]
print("1.b The determisnistic policy:")
print(mat)    
    
    
    

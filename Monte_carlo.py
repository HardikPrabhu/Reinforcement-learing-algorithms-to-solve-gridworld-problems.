import numpy
import random

#setting up the enviroment

def move(action, state):  #how  agent should move (not under the influence of the environment)
    if action == "L":
        return state[0], state[1] - 1
    if action == "R":
        return state[0], state[1] + 1
    if action == "U":
        return state[0] - 1, state[1]
    if action == "D":
        return state[0] + 1,state[1]

# given an input state and desired action returns new state, reward
def transition(action, state):
    #state is a (i,j) tuple, action is from [L,U,R,D]
    reward = 0

    #after reaching terminal state, no reward, no movement
    if state==(6,6):
        return (reward, state)

    a=["L","U","R","D"]
    val=a.index(action)  #clockwise == val+1, anticlock== val-1  (mod 4)
    next=numpy.random.choice(["success","stay","clk","ant-clk"], p=[0.8, 0.1, 0.05, 0.05])
    if next=="clk":
        val=(val+1)%4   # clockwise rotation
    if next=="ant-clk":
        val=(val-1)%4   # anticlockwise rotation
    if next!="stay":
     new_state=move(a[val],state)
    else:
        new_state=state

    if new_state[0]<1 or new_state[0]>6 or new_state[1]<1 or new_state[1]>6:
        new_state=state #invalid transition (hitting a wall)
    if new_state==(4,3) or new_state==(5,3):
        new_state=state #invalid transition (falling into the hole)

    if new_state==(6,3):
        reward=-15
    if new_state==(6,6):
        reward=15

    return(reward, new_state)

#CREATE POLICY DICT : policy[s]=[ p(L|s) , p(U|s) , p(R|s) , p(D|s) ]
def iniatize_policy_uniform():
 policy={}
 for i in range(1,7):
    for j in range(1,7):
        if (i,j) != (4, 3) and  (i,j)!= (5, 3):
              policy[(i,j)]=[0.25, 0.25, 0.25, 0.25]
 return policy 

policy= iniatize_policy_uniform()

Action=["L","U","R","D"]

# a function to randomly  Q values to zeros
def initialize_actionval(Q):
 for a in Action:
  for i in range(1,7):
      for j in range(1,7):
        Q[(i,j),a]=0
        
        
        
#generate an episode based on the given policy and starting state action pair
def gen_episode(state, action, policy):   
    rewards=[]      #[r1, r2,.....]
    states=[state]  #[s0, s1, s2, s3....]
    actions=[action] #[a0, a1, a2, a3,....]
    reward,state=transition(action,state)
    rewards.append(reward)
    states.append(state)
    count=0
    while state!=(6,6) and count<200:
     count=count+1
     action=numpy.random.choice(["L","U","R","D"], p=policy[state]) 
     actions.append(action)
     reward,state=transition(action,state)
     rewards.append(reward)
     states.append(state)
    episode={}
    state_action=[(states[i],actions[i]) for i in range(len(actions))]
    episode["rewards"]=rewards
    episode["state-action"]=state_action
    if state==(6,6):
      return episode
    else:
      return "exceeding number of timesteps"     
    
    
#lets define the policy improvement/update code
def update_policy(Q,policy): #deterministically choose optimal action at state s  
    for s in policy:
        argmax=random.choice(Action)
        max=Q[s,argmax]
        for a in Action:
            if Q[s,a]>max:
                argmax=a
                max=Q[s,a]
        prob=[0,0,0,0]
        index=Action.index(argmax)
        prob[index]=prob[index]+1
        policy[s]=prob
    return
  
  #Exploring start, randomly select inital s,a 
Action=["L","U","R","D"]
states=[] #all possible states allowed on the grid
for i in range(1,7):
    for j in range(1,7):
        if (i,j)!=(5,3) and (i,j)!=(4,3):
            states.append((i,j))
def starting():
   a=random.choice(Action)
   s=random.choice(states)
   return a,s

 def eval_episode(episode):
 if type(episode)!=str:
  G=episode["rewards"]
  SA=episode["state-action"]
  T=len(G)
  v=0
  Returns={}
  for i in range(T):
    t=T-i-1
    v=0.9*v+G[t]  #gamma=0.9 (dicount factor)
    Returns[SA[i]]=v
  return Returns

def MonteES(n):
  policy=iniatize_policy_uniform()     #uniform policy  
  Q={} #arbitrarily assign  state-action function dict
  initialize_actionval(Q)
  returnlist={}
  for i in Q:
    returnlist[i]=[]
  for i in range(n):
    print(i+1)
    a,s=starting()
    episode=gen_episode(s,a,policy)
    returns=eval_episode(episode)
    if returns:
     for j in returns: #j is a state-action pair
      returnlist[j].append(returns[j])
      #update the Q(j)
      Q[j]=sum(returnlist[j])/len(returnlist[j])
      update_policy(Q,policy)  
  return policy, returnlist

op_action={}
a=["L","U","R","D"]
for i in policy:
   if i!=(6,6):
    x=policy[i]
    for j in range(len(x)):
      if x[j]>0:
          op_action[i]=a[j]

#Monte carlo epsilon

def update_policy(e,Q,policy): #greedy update
    for s in policy:
        argmax=numpy.random.choice(["L", "U", "R", "D"],p= [0.25,0.25,0.25,0.25])
        max=Q[s,argmax]
        for a in Action:
            if Q[s,a]>max:
                argmax=a
                max=Q[s,a]
        prob=[e/4,e/4,e/4,e/4]
        index=Action.index(argmax)
        prob[index]=prob[index]+1-e
        policy[s]=prob
    return
  
  
  def Monte_epsilon(n):
  policy=iniatize_policy_uniform()     #uniform policy  
  Q={} #arbitrarily assign  state-action function dict
  initialize_actionval(Q)
  returnlist={}
  for i in Q:
    returnlist[i]=[]
  for i in range(n):
    print(i+1)
    a,s=starting()
    episode=gen_episode(s,a,policy)
    returns=eval_episode(episode)
    if returns:
     for j in returns: #j is a state-action pair
      returnlist[j].append(returns[j])
      #update the Q(j)
      Q[j]=sum(returnlist[j])/len(returnlist[j])
      update_policy(0.2,Q,policy)  
  return policy, returnlist
   

  

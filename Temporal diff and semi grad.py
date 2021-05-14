from matplotlib import pyplot as plt
import numpy
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

#CREATE POLICY DICT
policy={}
for i in range(1,7):
    for j in range(1,7):
        if (i,j) != (4, 3) and  (i,j)!= (5, 3):
              policy[(i,j)]=[0, 0, 0, 0]


Action=["L","U","R","D"]

# a function to randomly assign Q values
def initialize_actionval(Q):
 for a in Action:
  for i in range(1,7):
      for j in range(1,7):
          if (i, j) != (4, 3) and (i, j) != (5, 3) and (i,j)!=(6,6):
              Q[(i,j),a]=numpy.random.random()
          else:
              Q[(i,j),a]=0

#for example
Q={}
initialize_actionval(Q)

def update_policy(e,Q,policy):
    for s in policy:
        argmax="L"
        max=Q[s,"L"]
        for a in Action[1:]:
            if Q[s,a]>max:
                argmax=a
                max=Q[s,a]
        prob=[e/4,e/4,e/4,e/4]
        index=Action.index(argmax)
        prob[index]=prob[index]+1-e
        policy[s]=prob
    return



#Tabular method

#sarsa

def sarsa_episode(policy,Q,alpha,e):   #alpha is the step size e is epsilon for epsilon greedy policy used
    rewards=0
    s=(1,1) #start state
    while s!=(6,6):
     a=numpy.random.choice(["L", "U", "R", "D"], p=policy[s]) #choose action according to the policy
     r,s0=transition(a,s)
     rewards=rewards+r
     a0=numpy.random.choice(["L", "U", "R", "D"], p=policy[s0])
     Q[(s,a)]=Q[(s,a)]+alpha*(r+0.9*Q[(s0,a0)]-Q[(s,a)])
     s=s0
     update_policy(e,Q,policy)
    return rewards

def sarsa(episodes,alpha,e):
   # initialize a Q function
   initialize_actionval(Q)
   # based on it an e-greedy policy
   update_policy(e, Q, policy)

   rewards=[]
   for i in range(episodes):
       rewards.append(sarsa_episode(policy, Q, alpha, e))
   return (policy,rewards)


#Q-learning
def Qlearningepisode(policy,Q,alpha,e):
    s=(1,1)
    reward=0
    while s!=(6,6):
     a=numpy.random.choice(["L", "U", "R", "D"], p=policy[s])
     r, s0 = transition(a, s)
     reward=reward+r
     maxq=max([Q[s0,x] for x in Action])
     Q[s,a]=Q[s,a] + alpha*(r+0.9*maxq-Q[s,a])
     s=s0
     update_policy(e, Q, policy)
    return reward

def Qlearning(episodes,alpha,e):
    # initialize a Q function
    initialize_actionval(Q)
    # based on it an e-greedy policy
    update_policy(e, Q, policy)
    rewards = []
    for i in range(episodes):
        rewards.append(Qlearningepisode(policy, Q, alpha, e))
    return (policy, rewards)

episodes=numpy.arange(1,201)

#plots for tabular methods

rewards=numpy.zeros(200)
rewards2=numpy.zeros(200)
for i in range(1000):
    print(i)
    rewards=rewards+numpy.array(sarsa(200,0.2,0.1)[1]) #len 200 (reward per episode list)
    rewards2=rewards2+numpy.array(Qlearning(200,0.2,0.1)[1])
rewards=rewards/1000
rewards2=rewards2/1000
plt.figure(0)
plt.title("avg return for episodes (epsilon=0.1)")
plt.plot(episodes,rewards,label="Sarsa(tabular)")
plt.plot(episodes,rewards2,label="Qlearn(tabular)")
plt.legend()
plt.show()



#Linear Approximation (Semi-grad)
#There are total 144 possible combinations of state and actions
#We apply one hot encoding to have 144 dim vector to represent a state
l=list(Q.keys())
def encode(s,a):

   X=numpy.zeros(144)
   numpy.put(X,l.index((s,a)),1)
   return X


#q as a linear combination of weights and the encoding
def q(s,a,weights):
    return numpy.dot(weights,encode(s,a))

# the dictionary containing Q values for all (s,a)
def updateQ(weights):
    Q={}
    for a in Action:
        for i in range(1, 7):
            for j in range(1, 7):
                if (i, j) != (4, 3) and (i, j) != (5, 3):
                    Q[(i, j), a] = q((i,j),a,weights)
    return Q

#since Q values intitally will all be zero, old version of update function will give prirority to Left action, hence slow termination
#we will modify it.

def update_policy(e,Q,policy):
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



#sarsa -semi gradient
def sarsa_grad_episode(policy,alpha,e,weights):
    reward = 0
    s = (1, 1)  # start state
    while s != (6, 6):
        a = numpy.random.choice(["L", "U", "R", "D"], p=policy[s])  # choose action according to the policy
        r, s0 = transition(a, s)
        reward = reward + r
        a0 = numpy.random.choice(["L", "U", "R", "D"], p=policy[s0])
        delta=r+0.9*(weights[l.index((s0,a0))]-weights[l.index((s,a))])

        weights=weights+alpha*delta*encode(s,a)
        s=s0
        #update policy
        Q=updateQ(weights)
        update_policy(e,Q,policy)

    return reward,weights   #returns updated weight, reward after an episode


def sarsa_grad(episodes,alpha,e):
    #initialize weight vector
    w=numpy.zeros(144)
    #initialize Q values
    Q = updateQ(w)
    #initialize an e greedy policy based on it
    # initialize a Q function
    initialize_actionval(Q)
    # based on it an e-greedy policy #all values are zero because weights are zero initially
    # random e-greedy policy
    for s in policy:
        prob = [e / 4, e / 4, e / 4, e / 4]
        x = numpy.random.choice([0, 1, 2, 3], p=[0.25, 0.25, 0.25, 0.25])
        prob[x] = prob[x] + 1 - e
        policy[s] = prob
    rewards = []
    for i in range(episodes):
        reward,w=sarsa_grad_episode(policy,alpha, e,w)
        rewards.append(reward)
    return (policy, rewards)

#qlearning-semigradient
def qlearn_grad_episode(policy,alpha,e,weights):
    reward = 0
    s = (1, 1)  # start state
    while s != (6, 6):
        a = numpy.random.choice(["L", "U", "R", "D"], p=policy[s])  # choose action according to the policy
        r, s0 = transition(a, s)
        reward = reward + r
        Qhat=[q(s0,a,weights)for a in Action]
        delta = (r + 0.9 *(max(Qhat)-q(s,a,weights)))
        weights = weights + alpha * delta * encode(s, a)
        s=s0
        #update policy
        Q = updateQ(weights)
        update_policy(e, Q, policy)

    return reward,weights

def qlearn_grad(episodes,alpha,e):
    # initialize weight vector
    w = numpy.zeros(144)
    # initialize Q values
    Q = updateQ(w)
    # initialize a Q function
    initialize_actionval(Q)
    # based on it an e-greedy policy #all values are zero because weights are zero initially
    # random e-greedy policy
    for s in policy:
        prob = [e / 4, e / 4, e / 4, e / 4]
        x = numpy.random.choice([0, 1, 2, 3], p=[0.25, 0.25, 0.25, 0.25])
        prob[x] = prob[x] + 1 - e
        policy[s] = prob
    rewards = []
    for i in range(episodes):
        reward,w=qlearn_grad_episode(policy,alpha, e,w)
        rewards.append(reward)
    return (policy, rewards)



'''
#semi-gradient
rewards3=numpy.zeros(200)
rewards4=numpy.zeros(200)
for i in range(1000):
    print(i)
    r1=sarsa_grad(200, 0.2, 0.1)[1]
    rewards3=rewards3+numpy.array(r1)
    r2=qlearn_grad(200, 0.2, 0.1)[1]
    rewards4 = rewards4 + numpy.array(r2)
r3=rewards3/1000
r4=rewards4/1000

plt.figure(1)
plt.title("avg return for episodes (epsilon=0.05)")
plt.plot(episodes,r3,label="Sarsa(semi-grad)")
plt.plot(episodes,r4,label="Qlearn(semi-grad)")
plt.legend()
plt.show()'''

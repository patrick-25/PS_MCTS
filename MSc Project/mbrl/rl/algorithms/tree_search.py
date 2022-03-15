import numpy as np

# FIXME: test
class ChanceNode:
    def __init__(self, env, action, state, time_step, gamma, C):
        self.env = env
        
        self.action = action
        self.state = state
        self.time_step = time_step
        
        self.gamma = gamma
        self.C = C
        
        self.visits = 0
        self.acc_returns = 0.
        
        self.children = {}

    def average_return(self):
        if self.visits > 0:
            return self.acc_returns/self.visits
        
        return float('inf')
    
    def sample(self, max_steps, random_state):
        next_state, reward = self.env.draw(self.state, self.action)
        
        if next_state not in self.children:
            self.children[next_state] = DecisionNode(self.env, next_state, self.time_step + 1, self.gamma, self.C)
        
        ret = reward + self.gamma*self.children[next_state].sample(max_steps, random_state)
        
        self.visits += 1
        self.acc_returns += ret
        
        return ret
    
    
# FIXME: test
class DecisionNode:
    def __init__(self, env, state, time_step, gamma, C):
        # Note: Exploration guarantees require rewards in the interval [0, 1]
        self.env = env
        
        self.state = state
        self.time_step = time_step
        
        self.gamma = gamma
        self.C = C
        
        self.children = [ChanceNode(self.env, a, self.state, self.time_step, self.gamma, self.C) for a in range(self.env.n_actions)]
        
        self.visits = 0
        self.acc_returns = 0
        
    def sample(self, max_steps, random_state):
        if self.time_step == max_steps:
            ret = 0
        elif self.visits == 0:
            ret = self.rollout(max_steps, random_state)
        else:
            visits = np.array([n.visits for n in self.children])
            returns = np.array([n.average_return() for n in self.children])
            
            out = np.full(visits.shape, float('inf'))
            scores = np.divide(np.log(self.visits), visits, out=out, where=(visits > 0))
            
            returns = returns/(max_steps - self.time_step)
            
            scores = returns + self.C*np.sqrt(scores) 
            
            a = np.argmax(scores)
            ret = self.children[a].sample(max_steps, random_state)
        
        self.visits += 1
        self.acc_returns += ret
        
        return ret

    def rollout(self, max_steps, random_state):
        ret = 0.0
        g = 1.0
        
        state = self.state
        for _ in range(max_steps - self.time_step):
            action = random_state.choice(self.env.n_actions)
            
            state, reward = self.env.draw(state, action)
            
            ret += g*reward
            g *= self.gamma
        
        return ret
    
    def best_action(self):
        return np.argmax([n.average_return() for n in self.children])


# FIXME: test
def rho_uct(env, gamma, n_visits, C, seed):
    random_state = np.random.RandomState(seed)
    
    state = env.reset()
    node = DecisionNode(env, state, time_step=0, gamma=gamma, C=C)
    
    policy = {}
    value = {}
    
    done = False
    while not done:
        for _ in range(n_visits - node.visits):
            node.sample(env.max_steps, random_state)
           
        action = node.best_action()
        
        policy[state] = action
        value[state] = node.children[action].average_return()
        
        state, _, done = env.step(action)
        
        if state in node.children[action].children:
            node = node.children[action].children[state]
        else:
            node = DecisionNode(env, state, node.time_step + 1, gamma=gamma, C=C)
        
    return policy, value

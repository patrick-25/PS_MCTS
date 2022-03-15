import numpy as np

from environments.tabular import TabularEnvironmentModel
from algorithms.model_based import policy_iteration

from algorithms.tree_search import DecisionNode


class ModelLearner:
    def __init__(self, n_states, n_actions, alpha, rprior_variance, r_variance, seed):
        self.n_states = n_states
        self.n_actions = n_actions
        
        self.alpha = alpha
        self.rprior_precision = 1./rprior_variance
        self.r_precision = 1./r_variance
        
        self.random_state = np.random.RandomState(seed)
        
        self.N = np.zeros((n_states, n_states, n_actions))
        self.R = np.zeros((n_states, n_states, n_actions))
    
    def append(self, next_state, reward, state, action):
        self.N[next_state, state, action] += 1
        self.R[next_state, state, action] += reward

    def posterior_draw(self):
        p = np.zeros(self.N.shape, dtype=np.float)
        r = np.zeros(self.R.shape, dtype=np.float)
        
        for state in range(self.n_states):
            for action in range(self.n_actions):
                pcounts = self.alpha + self.N[:, state, action]
                p[:, state, action] = self.random_state.dirichlet(pcounts, 1)
                
                for next_state in range(self.n_states):
                    count = self.N[next_state, state, action]
                    
                    if count < 1:
                        rpost_precision = self.rprior_precision
                        rpost_mean = 0.
                    else:
                        rpost_precision = self.rprior_precision + count*self.r_precision
                        rpost_mean = (self.R[next_state, state, action]*self.r_precision)/(rpost_precision)
                        
                    r[next_state, state, action] = self.random_state.normal(rpost_mean, 1./rpost_precision)
         
        seed = self.random_state.randint(12)
        
        return TabularEnvironmentModel(self.n_states, self.n_actions, p, r, seed=seed)
    
    def maximum_likelihood(self):
        p = np.zeros(self.N.shape, dtype=np.float)
        r = np.zeros(self.R.shape, dtype=np.float)

        for state in range(self.n_states):
            for action in range(self.n_actions):
                if np.allclose(self.N[:, state, action], 0.):
                    ncounts = np.full(self.n_states, 1./self.n_states)
                else:
                    ncounts = self.N[:, state, action]
                    ncounts = ncounts/np.sum(ncounts)
                
                p[:, state, action] = ncounts
                
                for next_state in range(self.n_states):
                    count = self.N[next_state, state, action]
                    
                    if count < 1:
                        r[next_state, state, action] = 0.0
                    else:
                        ret = self.R[next_state, state, action]
                        r[next_state, state, action] = ret/count
        
        seed = self.random_state.randint(12)

        return TabularEnvironmentModel(self.n_states, self.n_actions, p, r, seed=seed)


def ps_policy_iteration(env, max_episodes, alpha, rprior_variance, r_variance, gamma, theta, max_iterations, seed):
    learner = ModelLearner(env.n_states, env.n_actions, alpha, rprior_variance, r_variance, seed)
    
    policy = None
    for _ in range(max_episodes):
        senv = learner.posterior_draw()
        # Use posterior sampling to choose (a likely) one of the possible environments - how to know which one's best??
        policy, value = policy_iteration(senv, gamma, theta, max_iterations, policy)
        # Use policy iteration to choose the optimal policy with its value
        # Here, we could instead do MCTS to find a good policy with its value
        state = env.reset()
        #^this part confuses me?
        
        done = False
        while not done:
            action = policy[state]
            next_state, reward, done = env.step(action)
            
            learner.append(next_state, reward, state, action)
            
            state = next_state
        
    senv = learner.maximum_likelihood()
    return policy_iteration(senv, gamma, theta, max_iterations, policy)

# Test:
def ps_rho_uct(env, max_episodes, alpha, rprior_variance, r_variance, gamma, n_visits, C, seed, verbose=False):
    learner = ModelLearner(env.n_states, env.n_actions, alpha, rprior_variance, r_variance, seed)
    
    policy = {}
    value = {}
    
    for t in range(max_episodes):
        senv = learner.posterior_draw()
        
        state = env.reset()
        node = DecisionNode(senv, state, time_step=0, gamma=gamma, C=C)
        
        done = False
        while not done:
            for _ in range(n_visits - node.visits):
                node.sample(env.max_steps, learner.random_state)
                
            action = node.best_action()
            
            policy[state] = action
            value[state] = node.children[action].average_return()
            
            next_state, reward, done = env.step(action)
            
            learner.append(next_state, reward, state, action)
            
            state = next_state
            
            if state in node.children[action].children:
                node = node.children[action].children[state]
            else:
                node = DecisionNode(senv, state, node.time_step + 1, gamma=gamma, C=C)
        
        if verbose:
            print('Episode {0}/{1}.'.format(t + 1, max_episodes))
            if hasattr(env, 'render_partial'):
                env.render_partial(policy, value)
                print('')
            
    return policy, value
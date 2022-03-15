import numpy as np
import contextlib


@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally: 
        np.set_printoptions(**original)


class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        
        self.random_state = np.random.RandomState(seed)
        
    def p(self, next_state, state, action):
        raise NotImplementedError()
    
    def r(self, next_state, state, action):
        raise NotImplementedError()
        
    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)
        
        return next_state, reward
    
    
class TabularEnvironmentModel(EnvironmentModel):
    def __init__(self, n_states, n_actions, p, r, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)
        
        self._p = p
        self._r = r

    def p(self, next_state, state, action):
        return self._p[next_state, state, action]

    def r(self, next_state, state, action):
        return self._r[next_state, state, action]
    
    
class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)
        
        self.max_steps = max_steps
        
        self.pi = pi
        if self.pi is None:
            self.pi = np.full(n_states, 1./n_states)
        
    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)
        
        return self.state
        
    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')
        
        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)
        
        self.state, reward = self.draw(self.state, action)
        
        return self.state, reward, done
    
    def render(self, policy=None, value=None):
        raise NotImplementedError()
        

class LinearWrapper:
    def __init__(self, env):
        self.env = env
        
        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states
        
    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0
          
        return features
    
    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)
        
        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)
            
            policy[s] = np.argmax(q)
            value[s] = np.max(q)
        
        return policy, value
        
    def reset(self):
        return self.encode_state(self.env.reset())
    
    def step(self, action):
        state, reward, done = self.env.step(action)
        
        return self.encode_state(state), reward, done
    
    def render(self, policy=None, value=None):
        self.env.render(policy, value)


class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)
        
        self.slip = slip
        
        n_states = self.lake.size + 1
        n_actions = 4
        
        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0
        
        self.absorbing_state = n_states - 1
        
        # up, left, down, right
        actions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        
        self._p = np.zeros((n_states, n_states, n_actions))
        
        self._p[self.absorbing_state, self.absorbing_state, :] = 1.0
        
        for s in range(self.absorbing_state):
            if (self.lake_flat[s] == '#') or (self.lake_flat[s] == '$'):
                self._p[self.absorbing_state, s, :] = 1.0
            else:
                neighbors = []
                for a in range(4):
                    c = np.unravel_index(s, self.lake.shape)
                    next_c = c[0] + actions[a][0], c[1] + actions[a][1]
                    next_s = np.ravel_multi_index(next_c, self.lake.shape, 'clip')
                    
                    self._p[next_s, s, a] = 1 - self.slip
                    
                    neighbors.append(next_s)
                    
                for neighbor in neighbors:
                    self._p[neighbor, s, :] += self.slip / n_actions
                
        Environment.__init__(self, n_states, n_actions, max_steps, pi, seed=seed)
        
    def p(self, next_state, state, action):
        return self._p[next_state, state, action]
    
    def r(self, next_state, state, action):
        if state < self.absorbing_state:
            return float(self.lake_flat[state] == '$')
        
        return 0.
    
    def step(self, action):
        state, reward, done = Environment.step(self, action)
        
        done = (state == self.absorbing_state) or done
        
        return state, reward, done
   
    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)
            
            if self.state < self.absorbing_state:
                lake[self.state] = '@'
                
            print(lake.reshape(self.lake.shape))
        else:
            actions = ['↑', '←', '↓', '→']
            
            print('Lake:')
            print(self.lake)
        
            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))
            
            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))
                
    # FIXME: implement based on the example in the class below
    def render_partial(self, policy, value):
        raise NotImplementedError()
          

# FIXME: test
class RiverSwim(Environment):
    def __init__(self, river, max_steps, seed):
        self.river = np.array(river)
        self.river_flat = self.river.reshape(-1)

        n_states = self.river.size + 1
        n_actions = 2
        pi = np.zeros(n_states, dtype=float)
        pi[0] = 1.0
        
        Environment.__init__(self, n_states, n_actions, max_steps, pi, seed)
        
    def p(self, next_state, state, action):
        if action == 0:
            return float(next_state == max(state - 1, 0))
        else:
            if next_state == state - 1:
                if state == self.n_states - 1:
                    return 0.4
                else:
                    return 0.05
            if next_state == state:
                if state == 0:
                    return 0.4
                else:
                    return 0.6
            if next_state == state + 1:
                if state == 0:
                    return 0.6
                else:
                    return 0.35
            
            return 0.
    
    def r(self, next_state, state, action):
        if action == 0:
            return 0.005*float(state == 0)
        else:
            if (state == self.n_states - 1) and (state == next_state):
                return 10.0
            if (state == 0) and (state == next_state):
                return 0.1
            else:
                return 0.0
            
    # FIXME: test
    def render(self, policy, value):
        if policy is None:
            river = np.array(self.river_flat)
            river[self.state] = '@'
            
            print(river.reshape(self.river.shape))
        else:
            actions = ['←', '→', '?']
            
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.river.shape))
            
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.river.shape))
    
    # FIXME: test
    def render_partial(self, policy, value):
        p = np.zeros(self.n_states, dtype=np.int)
        v = np.zeros(self.n_states, dtype=np.float)
        
        for s in range(self.n_states):
            p[s] = policy.get(s, self.n_actions)
            v[s] = value.get(s, float('nan'))
        
        self.render(p, v)

'''
        if policy is None:
            lake = np.array(self.lake_flat)
            
            if self.state < self.absorbing_state:
                lake[self.state] = '@'
                
            print(lake.reshape(self.lake.shape))
        else:
            actions = ['↑', '←', '↓', '→']
            
            print('Lake:')
            print(self.lake)
        
            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))
            
            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))'''
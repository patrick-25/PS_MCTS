import numpy as np


def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    for i in range(max_episodes):
        s = env.reset()
        
        if random_state.rand() < epsilon[i]:
            a = random_state.choice(env.n_actions)
        else:
            qmax = max(q[s])
            best = [a for a in range(env.n_actions) if np.allclose(qmax, q[s, a])]
            a = random_state.choice(best)
            
        done = False
        while not done:
            next_state, r, done = env.step(a)
            
            if random_state.rand() < epsilon[i]:
                next_action = random_state.choice(env.n_actions)
            else:
                qmax = max(q[next_state])
                best = [na for na in range(env.n_actions) if np.allclose(qmax, q[next_state, na])]
                next_action = random_state.choice(best)
            
            q[s, a] = q[s, a] + eta[i] * (r + gamma * q[next_state, next_action] - q[s, a])
            
            s = next_state
            a = next_action
        
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
        
    return policy, value


def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    for i in range(max_episodes):
        s = env.reset()
        
        done = False
        while not done:
            if random_state.rand() < epsilon[i]:
                a = random_state.choice(env.n_actions)
            else:
                qmax = max(q[s])
                best = [a for a in range(env.n_actions) if np.allclose(qmax, q[s, a])]
                a = random_state.choice(best)
                
            next_state, r, done = env.step(a)
            
            q[s, a] = q[s, a] + eta[i] * (r + gamma * max(q[next_state]) - q[s, a])
            
            s = next_state
        
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
        
    return policy, value


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    
    for i in range(max_episodes):
        features = env.reset()
        
        q = features.dot(theta)

        if random_state.rand() < epsilon[i]:
            a = random_state.choice(env.n_actions)
        else:
            qmax = max(q)
            best = [a for a in range(env.n_actions) if np.allclose(qmax, q[a])]
            a = random_state.choice(best)
            
        done = False
        while not done:
            next_features, r, done = env.step(a)
            
            next_q = next_features.dot(theta)
            
            if random_state.rand() < epsilon[i]:
                next_action = random_state.choice(env.n_actions)
            else:
                qmax = max(next_q)
                best = [na for na in range(env.n_actions) if np.allclose(qmax, next_q[na])]
                next_action = random_state.choice(best)
                
            td = r + gamma*next_q[next_action] - q[a]
            theta = theta + eta[i]*td*features[a]
            
            features = next_features
            q = features.dot(theta)
            a = next_action
    
    return theta


def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    
    for i in range(max_episodes):
        features = env.reset()
        
        done = False
        while not done:
            q = features.dot(theta)
            
            if random_state.rand() < epsilon[i]:
                a = random_state.choice(env.n_actions)
            else:
                qmax = max(q)
                best = [a for a in range(env.n_actions) if np.allclose(qmax, q[a])]
                a = random_state.choice(best)
                
            next_features, r, done = env.step(a)
            
            next_q = next_features.dot(theta)
            
            td = r + gamma*max(next_q) - q[a]
            
            theta = theta + eta[i]*td*features[a]
            
            features = next_features

    return theta    

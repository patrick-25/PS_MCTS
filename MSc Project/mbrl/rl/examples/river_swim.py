from rl.environments.tabular import FrozenLake, RiverSwim
from rl.environments.tabular import LinearWrapper

from rl.algorithms.model_based import policy_iteration
from rl.algorithms.model_based import value_iteration

from rl.algorithms.model_free import sarsa
from rl.algorithms.model_free import q_learning

from rl.algorithms.model_free import linear_sarsa
from rl.algorithms.model_free import linear_q_learning

from rl.algorithms.model_building import ps_policy_iteration


def play(env):
    actions = ['a', 'd']

    state = env.reset()
    env.render()

    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            raise Exception('Invalid action')

        state, r, done = env.step(actions.index(c))

        env.render()
        print('Reward: {0}.'.format(r))


def main():
    seed = 0



    # flat river
    river = [['&', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '$']]
    # make moving in & have reward 5
    # make moving in $ have reward 10000
    env = RiverSwim(river, max_steps=100, seed=seed)

    print('# Model-based algorithms')
    gamma = 0.9
    theta = 0.001
    max_iterations = 500

    print('')

    print('## Policy iteration')
    policy, value = policy_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    print('')

    print('## Value iteration')
    policy, value = value_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    print('')

    print('# Model-free algorithms')
    max_episodes = 2000
    eta = 0.5
    epsilon = 0.5

    print('')
    '''

    print('## Sarsa')
    policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)

    print('')

    print('## Q-learning')
    policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)

    print('')

    linear_env = LinearWrapper(env)

    print('## Linear Sarsa')

    parameters = linear_sarsa(linear_env, max_episodes, eta, gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

    print('')

    print('## Linear Q-learning')

    parameters = linear_q_learning(linear_env, max_episodes, eta, gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)
    '''
    print('')

    print('# Model-building algorithm')
    max_episodes = 200
    alpha = 0.01
    rprior_variance = 2
    r_variance = 1e-8

    print('')

    print('## Posterior sampling policy iteration')
    policy, value = ps_policy_iteration(env, max_episodes, alpha, rprior_variance, r_variance, gamma, theta,
                                        max_iterations, seed=seed)
    env.render(policy, value)

    print('')


if __name__ == "__main__":
    main()
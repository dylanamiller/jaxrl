import numpy as np
import jax.numpy as jnp
from sklearn.kernel_approximation import RBFSampler


class UCB:
    """ think about if i need n_agents in coverage. if so, need correct shaping
        i don't think i do: vectorized agents are represented by same policy
    """

    def __init__(self, env, agent, rbf_dim, cover_decay=0.7, threshold=1.0, lam=1.0):
        self.num_actions = env.action_spaces[agent].n
        self.feature_dim = env.observation_spaces[agent].shape[0] + self.num_actions
        self.rbf_dim = rbf_dim

        self.coverage = np.zeros((self.rbf_dim, self.rbf_dim))
        self.conver_decay = cover_decay
        self.threshold = threshold
        self.lam = lam # 1e-3

        self.rbf_feature = RBFSampler(gamma=1, random_state=1, n_components=self.rbf_dim)
        self.rbf_feature.fit(X = np.random.randn(5, self.feature_dim))

    def sigma_n(self, features):
        """ line 4 of PCPG: compute covariance matrix from trajectory features -> policy cover for current policy
        """

        cov = np.zeros_like(self.coverage)
        for feature in features:
            feature = np.expand_dims(feature, axis=1)
            cov += np.dot(feature, feature.T)
        return cov

    def compute_bonus(self, features, i_policy):
        """ compute cumulative policy cover, invert it, and approximated eigenvalues for exploration bonuses
        """

        self.coverage = self.conver_decay*self.coverage + self.sigma_n(features)
        cov_reg = self.coverage / i_policy + self.lam*np.eye(self.rbf_dim)     # coverage regularized
        # may want to try np.cholesky_inverse
        cov_inv = np.linalg.inv(cov_reg)                                               # coverage inverse
        bonuses = np.zeros(features.shape[0])
        for i, feature in enumerate(features):
            feature = np.expand_dims(feature, axis=1)
            bonus = np.dot(np.dot(feature.T, cov_inv), feature)
            if bonus > self.threshold:
                bonuses[i] = 5
            else:
                bonuses[i] = bonus

        return jnp.array(bonuses)

    def process_features(self, observations, actions):
        """ need to concatenate, process with rbf kernel (sklearn), return features
        """

        one_hot_actions = np.zeros((actions.size, actions.max()+1))
        one_hot_actions[np.arange(actions.size),actions] = 1
        obs_act_cat = np.concatenate((observations, one_hot_actions.reshape(-1, self.num_actions)), axis = -1)
        return self.rbf_feature.transform(obs_act_cat)



class TruncatedUCB:
    """ think about if i need n_agents in coverage. if so, need correct shaping
        i don't think i do: vectorized agents are represented by same policy
    """

    def __init__(self, env, agent, rbf_dim, cover_decay=0.7, threshold=1.0, lam=1.0):
        self.num_actions = env.action_spaces[agent].n
        self.feature_dim = env.observation_spaces[agent].shape[0] + self.num_actions
        self.rbf_dim = rbf_dim

        self.coverage = np.zeros((self.rbf_dim, self.rbf_dim))
        self.conver_decay = cover_decay
        self.threshold = threshold
        self.lam = lam # 1e-3

        self.rbf_feature = RBFSampler(gamma=1, random_state=1, n_components=self.rbf_dim)
        self.rbf_feature.fit(X = np.random.randn(5, self.feature_dim))

    def sigma_n(self, features):
        """ line 4 of PCPG: compute covariance matrix from trajectory features -> policy cover for current policy
        """

        cov = np.zeros_like(self.coverage)
        for feature in features:
            feature = np.expand_dims(feature, axis=1)
            cov += np.dot(feature, feature.T)
        return cov

    def compute_bonus(self, features, i_policy):
        """ compute cumulative policy cover, invert it, and approximated eigenvalues for exploration bonuses
        """

        self.coverage = self.conver_decay*self.coverage + self.sigma_n(features)
        cov_reg = self.coverage / i_policy + self.lam*np.eye(self.rbf_dim)     # coverage regularized
        # may want to try np.cholesky_inverse
        cov_inv = np.linalg.inv(cov_reg)                                               # coverage inverse
        bonuses = np.zeros(features.shape[0])
        for i, feature in enumerate(features):
            feature = np.expand_dims(feature, axis=1)
            bonus = np.dot(np.dot(feature.T, cov_inv), feature)
            if bonus > self.threshold:
                bonuses[i] = 5
            else:
                bonuses[i] = bonus

        return jnp.array(bonuses)

    def process_features(self, observations, actions):
        """ need to concatenate, process with rbf kernel (sklearn), return features
        """

        one_hot_actions = np.zeros((actions.size, actions.max()+1))
        one_hot_actions[np.arange(actions.size),actions] = 1
        obs_act_cat = np.concatenate((observations, one_hot_actions.reshape(-1, self.num_actions)), axis = -1)
        return self.rbf_feature.transform(obs_act_cat)
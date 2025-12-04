import numpy as np


# =========================
#   Configuration / Setup
# =========================

# We work with 7 states indexed 0..6.
states = np.arange(0, 7)

# By convention, state 6 is the "bottom" state in the Baird-style example.
lower_state = 6

# Discount factor γ
discount = 0.99

# Each state is mapped to an 8-dimensional feature vector x(s)
feature_vector_size = 8

# Feature matrix Φ of shape [num_states, feature_vector_size]
features = np.zeros((len(states), feature_vector_size))

# Construct features for the 6 upper states (0..5)
for i in range(len(states) - 1):
    # Give each upper state its own "active" feature with value 2
    features[i, i] = 2.0
    # All upper states share the last feature with value 1
    features[i, feature_vector_size - 1] = 1.0

# For the lower state: second-to-last feature = 1, last feature = 2
features[lower_state, feature_vector_size - 2] = 1.0
features[lower_state, feature_vector_size - 1] = 2.0

# Action identifiers
actions = {"dashed": 0, "solid": 1}

# Reward is always zero in this counterexample
reward = 0.0

# Under the behavior policy, the solid action is chosen with probability 1/7
behavior_solid_probability = 1.0 / 7.0

# Behavior-state distribution μ: uniform over all 7 states
state_distribution = np.ones(len(states)) / len(states)
state_distribution_matrix = np.diag(state_distribution)

# Projection matrix Π that projects onto the feature space under μ
projection_matrix = (
    features
    @ np.linalg.pinv(features.T @ state_distribution_matrix @ features)
    @ features.T
    @ state_distribution_matrix
)

# Interest i(s) is constant 1 for all states
interest = 1.0


# =========================
#   Environment dynamics
# =========================

def step(state, action):
    """
    One step of the MDP dynamics.

    Parameters
    ----------
    state : int
        Current state index.
    action : int
        Either actions["solid"] or actions["dashed"].

    Returns
    -------
    int
        Next state index.
    """
    # Solid action: always jump to the lower state
    if action == actions["solid"]:
        return lower_state

    # Dashed action: move to one of the upper states, chosen uniformly
    return np.random.choice(states[:lower_state])


# =========================
#         Policies
# =========================

def target_policy(state):
    """
    Target policy π: always selects the solid action.

    Parameters
    ----------
    state : int
        Current state (unused, since the policy is deterministic).

    Returns
    -------
    int
        Action index for the target policy.
    """
    # The target policy is deterministic: always choose "solid".
    return actions["solid"]


def behavior_policy(state):
    """
    Behavior policy μ: occasionally takes the solid action.

    Parameters
    ----------
    state : int
        Current state (not needed since behavior is state-independent).

    Returns
    -------
    int
        Sampled action index under the behavior policy.
    """
    # With probability 1/7, select the solid action; otherwise dashed.
    if np.random.binomial(n=1, p=behavior_solid_probability) == 1:
        return actions["solid"]
    return actions["dashed"]


# =========================
#    Error metrics
# =========================

def compute_RMSVE(weights):
    """
    Root Mean Squared Value Error (RMSVE) for a linear value function.

    Parameters
    ----------
    weights : np.ndarray
        Parameter vector w for the linear value function v_hat(s) = w^T x(s).

    Returns
    -------
    float
        RMSVE under the state distribution μ.
    """
    # Approximate value for each state: v_hat(s) = Φ_s · w
    approx_values = features @ weights

    # In this setup, true values v_π(s) are zero, so the error is v_hat(s)^2
    squared_errors = approx_values ** 2

    # Mean squared value error under μ
    ms_ve = np.dot(squared_errors, state_distribution)

    # Root of MSVE
    return np.sqrt(ms_ve)


def compute_RMSPBE(weights):
    """
    Root Mean Squared Projected Bellman Error (RMSPBE).

    Parameters
    ----------
    weights : np.ndarray
        Parameter vector w for the linear value function.

    Returns
    -------
    float
        RMSPBE under the state distribution μ.
    """
    # Bellman error δ(s) for each state (initialized to zero)
    bellman_errors = np.zeros(len(states))

    # For this MDP, transitions only matter when the next state is the lower one.
    # We accumulate the contribution for each state -> lower_state transition.
    for s in states:
        for next_state in states:
            if next_state == lower_state:
                bellman_errors[s] += (
                    reward
                    + discount * np.dot(weights, features[next_state, :])
                    - np.dot(weights, features[s, :])
                )

    # Project Bellman error back into the feature space using Π
    projected_be = projection_matrix @ bellman_errors

    # True value is zero, so the squared error is just projected_be^2
    squared_errors = projected_be ** 2

    # Mean squared projected Bellman error under μ
    ms_pbe = np.dot(squared_errors, state_distribution)

    # Root of MSPBE
    return np.sqrt(ms_pbe)


# =========================
#  Semi-gradient methods
# =========================

def semi_gradient_off_policy_TD(state, weights, step_size):
    """
    One step of semi-gradient off-policy TD(0).

    Parameters
    ----------
    state : int
        Current state.
    weights : np.ndarray
        Parameter vector w (updated in-place).
    step_size : float
        Learning rate α.

    Returns
    -------
    int
        Next state after taking an action from the behavior policy.
    """
    # Sample from the behavior policy
    action = behavior_policy(state)

    # Environment transition
    next_state = step(state, action)

    # Importance sampling ratio ρ = π(a|s) / μ(a|s)
    if action == actions["dashed"]:
        rho = 0.0
    else:
        rho = 1.0 / behavior_solid_probability

    # TD error δ = r + γ v_hat(s') - v_hat(s)
    td_error = (
        reward
        + discount * np.dot(features[next_state, :], weights)
        - np.dot(features[state, :], weights)
    )

    # Semi-gradient TD update
    weights += step_size * rho * td_error * features[state, :]

    return next_state


def semi_gradient_DP(weights, step_size):
    """
    Semi-gradient DP-style update using expected transitions.

    Parameters
    ----------
    weights : np.ndarray
        Parameter vector w (updated in-place).
    step_size : float
        Learning rate α.
    """
    # Accumulate gradient over all states
    accumulated_gradient = np.zeros_like(weights, dtype=float)

    for s in states:
        expected_return = 0.0

        # Only transitions to lower_state contribute in this setup
        for next_state in states:
            if next_state == lower_state:
                expected_return += reward + discount * np.dot(
                    weights, features[next_state, :]
                )

        # Bellman error for this state
        bellman_error = expected_return - np.dot(weights, features[s, :])

        # Gradient contribution for this state
        accumulated_gradient += bellman_error * features[s, :]

    # Average over states and apply semi-gradient update
    weights += (step_size / len(states)) * accumulated_gradient


# =========================
#     Gradient-TD family
# =========================

def TDC(state, weights, LLS_solution, step_size_w, step_size_v):
    """
    TDC / GTD(0) update for one sampled transition.

    Parameters
    ----------
    state : int
        Current state.
    weights : np.ndarray
        Main parameter vector w (updated in-place).
    LLS_solution : np.ndarray
        Secondary parameter vector v (updated in-place).
    step_size_w : float
        Learning rate for w (α).
    step_size_v : float
        Learning rate for v (β).

    Returns
    -------
    int
        Next state.
    """
    # Choose action under behavior policy
    action = behavior_policy(state)

    # Environment transition
    next_state = step(state, action)

    # Importance sampling ratio ρ
    if action == actions["dashed"]:
        rho = 0.0
    else:
        rho = 1.0 / behavior_solid_probability

    # TD error δ
    td_error = (
        reward
        + discount * np.dot(features[next_state, :], weights)
        - np.dot(features[state, :], weights)
    )

    phi_s = features[state, :]
    phi_next = features[next_state, :]

    # Update rule for w
    weights += step_size_w * rho * (
        td_error * phi_s - discount * phi_next * np.dot(phi_s, LLS_solution)
    )

    # Update rule for v (LMS-style update)
    LLS_solution += step_size_v * rho * (td_error - np.dot(phi_s, LLS_solution)) * phi_s

    return next_state


def expected_TDC(weights, LLS_solution, step_size_w, step_size_v):
    """
    Expected version of TDC using analytical expectations.

    Parameters
    ----------
    weights : np.ndarray
        Main parameter vector w (updated in-place).
    LLS_solution : np.ndarray
        Secondary parameter vector v (updated in-place).
    step_size_w : float
        Learning rate for w (α).
    step_size_v : float
        Learning rate for v (β).
    """
    for s in states:
        # For the expected update, we assume the next state is the lower state
        td_error = (
            reward
            + discount * np.dot(features[lower_state, :], weights)
            - np.dot(features[s, :], weights)
        )

        # Importance sampling ratio when action is solid
        rho = 1.0 / behavior_solid_probability

        # State probability under μ is uniform
        state_prob = 1.0 / len(states)

        # Expected update for w
        expected_update_w = (
            state_prob
            * behavior_solid_probability
            * rho
            * (
                td_error * features[s, :]
                - discount * features[lower_state, :] * np.dot(LLS_solution, features[s, :])
            )
        )

        weights += step_size_w * expected_update_w

        # Expected update for v
        expected_update_v = (
            state_prob
            * behavior_solid_probability
            * rho
            * (td_error * np.dot(LLS_solution, features[s, :]))
            * features[lower_state, :]
        )

        LLS_solution += step_size_v * expected_update_v


# =========================
#     Emphatic TD (ETD)
# =========================

def expected_emphatic_TD(weights, emphasis, step_size):
    """
    Expected Emphatic TD update, synchronously adjusting weights and emphasis.

    Parameters
    ----------
    weights : np.ndarray
        Parameter vector w (updated in-place).
    emphasis : float
        Current emphasis M.
    step_size : float
        Learning rate α.

    Returns
    -------
    float
        Expected emphasis value for the next step (averaged over states).
    """
    # Aggregate effect over all states
    total_update = 0.0
    total_next_emphasis = 0.0

    for s in states:
        # Importance sampling ratio for the lower state under the target policy
        if s == lower_state:
            rho = 1.0 / behavior_solid_probability
        else:
            rho = 0.0

        # Emphasis update: M_{t+1} = γ ρ_t M_t + i(s)
        next_emphasis = discount * rho * emphasis + interest
        total_next_emphasis += next_emphasis

        # For expected update, assume next state is always lower_state
        td_error = (
            reward
            + discount * np.dot(features[lower_state, :], weights)
            - np.dot(features[s, :], weights)
        )

        # State probability is uniform; action probability is behavior_solid_probability
        state_prob = 1.0 / len(states)
        total_update += (
            state_prob
            * behavior_solid_probability
            * next_emphasis
            * rho
            * td_error
            * features[lower_state, :]
        )

    # Update the weights with the accumulated expected update
    weights += step_size * total_update

    # Return average next emphasis across states
    return total_next_emphasis / len(states)

import numpy as np
from src.tile_coding import IHT, tiles


# ===================================================================== #
#  Environment & Action Definitions
# ===================================================================== #

ACTIONS = {
    "reverse": -1,
    "neutral": 0,
    "forward": 1,
}

POSITION = {"min": -1.2, "max": 0.5}
VELOCITY = {"min": -0.07, "max": 0.07}

# Optimistic initialization → ε = 0
EPSILON = 0.0


# ===================================================================== #
#  Policy & Dynamics
# ===================================================================== #

def get_action(position: float, velocity: float, value_function):
    """
    Select an action using an ε-greedy policy with respect to the
    current action-value estimates.

    Parameters
    ----------
    position : float
    velocity : float
    value_function : ValueFunction

    Returns
    -------
    int
        Action: -1, 0, or 1
    """

    # Exploration (rare with ε=0, but kept for algorithm completeness)
    if np.random.binomial(1, EPSILON):
        return np.random.choice(list(ACTIONS.values()))

    # Greedy selection based on Q(s,a)
    q_values = [
        value_function.value(position, velocity, action)
        for action in ACTIONS.values()
    ]

    # Pick randomly among ties
    max_value = np.max(q_values)
    greedy_actions = [
        action for action, q in zip(ACTIONS.values(), q_values) if q == max_value
    ]
    return np.random.choice(greedy_actions)


def step(position: float, velocity: float, action: int):
    """
    Simulate one environment transition for Mountain Car.

    Returns
    -------
    new_position : float
    new_velocity : float
    reward : float
    """

    # Velocity update
    new_velocity = velocity + 0.001 * action - 0.0025 * np.cos(3 * position)
    new_velocity = np.clip(new_velocity, VELOCITY["min"], VELOCITY["max"])

    # Position update
    new_position = position + new_velocity
    new_position = np.clip(new_position, POSITION["min"], POSITION["max"])

    # Velocity reset on left boundary
    if new_position == POSITION["min"]:
        new_velocity = 0.0

    # Every transition costs -1
    return new_position, new_velocity, -1.0


# ===================================================================== #
#  Value Function with Tile Coding Approximation
# ===================================================================== #

class ValueFunction:
    """
    Linear value function approximator using tile coding.
    """

    def __init__(self, step_size: float, num_tilings: int = 8, max_size: int = 2048):
        self.alpha = step_size / num_tilings
        self.num_tilings = num_tilings

        self.max_size = max_size
        self.hash_table = IHT(max_size)
        self.weights = np.zeros(max_size)

        # Scale continuous variables to tile-coding input range
        self.position_scale = num_tilings / (POSITION["max"] - POSITION["min"])
        self.velocity_scale = num_tilings / (VELOCITY["max"] - VELOCITY["min"])

    # ------------------------------------------------------------------ #

    def get_active_tiles(self, position, velocity, action):
        """Return tile indices active for (position, velocity, action)."""

        scaled_position = self.position_scale * position
        scaled_velocity = self.velocity_scale * velocity

        return tiles(
            iht_or_size=self.hash_table,
            num_tilings=self.num_tilings,
            floats=[scaled_position, scaled_velocity],
            ints=[action],
        )

    # ------------------------------------------------------------------ #

    def value(self, position, velocity, action):
        """
        Compute Q(s,a) using the current weights.
        Terminal states return 0 by definition.
        """
        if position == POSITION["max"]:
            return 0.0

        active_tiles = self.get_active_tiles(position, velocity, action)
        return np.sum(self.weights[active_tiles])

    # ------------------------------------------------------------------ #

    def learn(self, position, velocity, action, target):
        """
        Semi-gradient update of the value function using the TD target.
        """
        active_tiles = self.get_active_tiles(position, velocity, action)
        estimation = np.sum(self.weights[active_tiles])

        delta = self.alpha * (target - estimation)

        for tile_idx in active_tiles:
            self.weights[tile_idx] += delta

    # ------------------------------------------------------------------ #

    def cost_to_go(self, position, velocity):
        """
        Compute cost-to-go estimate V(s) = -max_a Q(s,a).
        """
        q_values = [
            self.value(position, velocity, action)
            for action in ACTIONS.values()
        ]
        return -np.max(q_values)


# ===================================================================== #
#  Semi-Gradient n-step SARSA
# ===================================================================== #

def semi_gradient_n_step_sarsa(value_function: ValueFunction, n: int = 1):
    """
    Run one episode of semi-gradient n-step SARSA.

    Parameters
    ----------
    value_function : ValueFunction
    n : int
        Number of steps for n-step SARSA.

    Returns
    -------
    int
        Number of time steps in the episode.
    """

    # Initial state
    position = np.random.uniform(-0.6, -0.4)
    velocity = 0.0
    action = get_action(position, velocity, value_function)

    positions = [position]
    velocities = [velocity]
    actions = [action]
    rewards = [0.0]

    time_step = 0
    episode_end = float("inf")

    while True:
        time_step += 1

        # If episode is ongoing, generate transition
        if time_step < episode_end:
            new_pos, new_vel, reward = step(position, velocity, action)
            new_act = get_action(new_pos, new_vel, value_function)

            positions.append(new_pos)
            velocities.append(new_vel)
            actions.append(new_act)
            rewards.append(reward)

            if new_pos == POSITION["max"]:
                episode_end = time_step

        # Determine which timestep to update
        update_t = time_step - n

        if update_t >= 0:

            # Compute n-step return
            G = 0.0

            # Add rewards
            upper_bound = min(episode_end, update_t + n)
            for t in range(update_t + 1, upper_bound + 1):
                G += rewards[t]

            # Bootstrap if episode not done
            if update_t + n <= episode_end:
                G += value_function.value(
                    positions[update_t + n],
                    velocities[update_t + n],
                    actions[update_t + n],
                )

            # Update only non-terminal states
            if positions[update_t] != POSITION["max"]:
                value_function.learn(
                    positions[update_t],
                    velocities[update_t],
                    actions[update_t],
                    G,
                )

        # Termination of episode once last update is applied
        if update_t == episode_end - 1:
            break

        # Move to next transition
        position = new_pos
        velocity = new_vel
        action = new_act

    return time_step


# ===================================================================== #
#  Visualization Helper
# ===================================================================== #

def print_cost(value_function: ValueFunction, episode: int, ax):
    """
    Plot the learned cost-to-go function V(s) over a grid of
    positions and velocities.
    """

    grid = 40
    positions = np.linspace(POSITION["min"], POSITION["max"], grid)
    velocities = np.linspace(VELOCITY["min"], VELOCITY["max"], grid)

    xs, ys, zs = [], [], []

    for p in positions:
        for v in velocities:
            xs.append(p)
            ys.append(v)
            zs.append(value_function.cost_to_go(p, v))

    ax.scatter(xs, ys, zs)
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_zlabel("Cost-to-Go")
    ax.set_title(f"Episode {episode + 1}")

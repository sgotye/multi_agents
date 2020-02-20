import numpy as np
import os


class ParticlesSwarm(object):
    """
    The ParticlesSwarm object is a model of swarm of agents in the environment.

    Attributes:
        n (int): total number of agents
        with_leader (bool): if there is a leader in the swarm
        n_agents (int): the number of agents for which dissipations are computed.
                              If there is a leader, its dissipation doesn't matter.
    """
    def __init__(self, n, with_leader=False):
        # TODO move constants to settings file
        self.n = n
        self.with_leader = with_leader
        if with_leader:
            self.leader_velocity = np.array([0.0, 0.1])
            self.n_agents = n - 1
        else:
            self.n_agents = n

        # TODO make it better
        self.space_dim = os.environ.get("DIMENSION_OF_SPACE", 2)
        self.bins_num = os.environ.get("SINGLE_AXIS_BINS", 11)
        self.distance_to_leader = os.environ.get("DISTANCE_TO_LEADER", 3)
        self._r_comfort_zone = os.environ.get("COMFORT_ZONE_RADIUS", 0.9)
        self._max_force = os.environ.get("MAX_ABS_FORCE", 0.3)
        self._a = os.environ.get("A_VALUE", 0.1)
        self._eta = os.environ.get("ETA_VALUE", 0.985)
        self._last_positions = None
        self._last_velocities = None
        self._zetas = None

    def reset(self):
        """
        Resets environment to the initial state.
        Always call before start.
        """
        self._last_positions = [
            np.random.rand(self.space_dim) for _ in range(self.n)]
        self._update_zetas()
        self._last_velocities = [
            np.array([0.0, 0.1]) for _ in range(self.n_agents)]
        if self.with_leader:
            self._last_velocities += [self.leader_velocity]
        init_state = self._get_init_state()
        # init_state = [(0, 1) for _ in range(self.n)]
        init_state = self._transform_state(init_state)
        return init_state

    def step(self, action):
        F_agent = self._convert_action_to_force(action)
        F_ext = [-f for f in F_agent]  # stationary case
        self._compute_velocities(F_ext)

        closest_agents = self._find_the_closest()
        in_comfort_zone = self._check_closest_in_comfort_zone(closest_agents)
        resides_closest = self._check_resides_closest(closest_agents)
        in_direction = self._check_in_direction()
        dissipations = self._compute_dissipations(F_agent, F_ext)
        rewards = self._get_rewards(
            in_comfort_zone, resides_closest, in_direction, dissipations)
        new_state = zip(in_comfort_zone, resides_closest, in_direction)
        new_state = self._transform_state(new_state)
        self._update_positions()
        self._update_zetas()
        return new_state, rewards, dissipations

    def get_last_positions(self):
        return self._last_positions

    def get_last_velocities(self):
        return self._last_velocities

    def _update_zetas(self):
        zetas = np.zeros((self.n, self.n, self.space_dim, self.space_dim))
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    r_i = np.array(self._last_positions[i])
                    r_j = np.array(self._last_positions[j])
                    R_ij = np.sqrt(np.dot(r_i - r_j, r_i - r_j))
                    R_hat_squared = (np.outer(r_i - r_j, r_i - r_j) /
                                     (R_ij ** 2))
                    Ident = np.identity(self.space_dim)

                    zetas[i][j] = ((0.75 * (self._a / R_ij) *
                                   (Ident + R_hat_squared) +
                                   0.5 * ((self._a / R_ij)**3) *
                                   (Ident - 3 * R_hat_squared)) /
                                   (6 * np.pi * self._eta * self._a))
                else:
                    Ident = np.identity(self.space_dim)
                    zetas[i][i] = Ident / (6 * np.pi * self._eta * self._a)
        self._zetas = zetas

    def _compute_velocities(self, F):
        force_for_compute = F
        if self.with_leader:
            leader_force = self._compute_force_for_leader(F)
            force_for_compute.append(leader_force)

        force_for_compute = np.array(force_for_compute)
        self._last_velocities = list(- np.tensordot(
            force_for_compute, self._zetas[:self.n_agents,:,:,:], axes=((0,1), (1,3))))
        if self.with_leader:
            self._last_velocities += [self.leader_velocity]

    def _compute_force_for_leader(self, F_ext):
        zetas_for_force = self._zetas[self.n - 1, :(self.n - 1), :, :]
        sum_zetas_F = np.zeros(self.space_dim)
        for agent, agent_force in enumerate(F_ext):
            sum_zetas_F += np.dot(zetas_for_force[agent], agent_force)

        F_ext_leader = -(self.leader_velocity +
                         sum_zetas_F) / self._zetas[
                             (self.n - 1), (self.n - 1), 0, 0]
        return F_ext_leader

    def _update_positions(self):
        for agent, velocity in enumerate(self._last_velocities):
            self._last_positions[agent] += velocity

    def _find_the_closest(self):
        closest_agents = []
        for i in range(self.n_agents):
            r_i = self._last_positions[i]
            distances_i = []
            for j in range(self.n):
                if j != i:
                    r_j = self._last_positions[j]
                    distances_i.append(np.dot(r_j - r_i, r_j - r_i))
                else:
                    distances_i.append(10**10)
            closest_agents.append(np.argmin(distances_i))
        return closest_agents

    def _check_closest_in_comfort_zone(self, closest_agents):
        in_comfort_zone = []
        for i, closest_agent in enumerate(closest_agents):
            r_i = self._last_positions[i]
            r_closest_i = self._last_positions[closest_agent]
            dist_to_closest_i = np.sqrt(np.dot(
                r_closest_i - r_i, r_closest_i - r_i))
            in_comfort_zone.append(np.heaviside(
                self._r_comfort_zone - dist_to_closest_i, 1))
        return in_comfort_zone

    def _compute_dissipations(self, F_agent, F_ext):
        dissipations = []
        for i in range(self.n_agents):
            P_i = - (np.dot(F_agent[i], F_ext[i]) /
                     (6 * np.pi * self._eta * self._a))
            dissipations.append(P_i)
        return dissipations

    def _check_resides_closest(self, closest_agents):
        resides_closest = []
        for i, closest_idx in enumerate(closest_agents):
            r_closest_i = self._last_positions[closest_idx]
            r_i = self._last_positions[i]
            initial_distance = np.sqrt(np.dot(r_i - r_closest_i, r_i - r_closest_i))

            v_i = self._last_velocities[i]
            v_closest_i = self._last_velocities[closest_idx]

            next_position = (r_i + v_i) - (r_closest_i + v_closest_i)
            next_distance = np.sqrt(np.dot(next_position, next_position))
            resides_closest.append(np.heaviside(-next_distance + initial_distance, 1))
        return resides_closest

    def _check_resides_closest_old(self, closest_agents):
        resides_closest = []
        for i, closest_idx in enumerate(closest_agents):
            r_closest_i = self._last_positions[closest_idx]
            r_i = self._last_positions[i]
            initial_distance = np.sqrt(np.dot(r_i - r_closest_i, r_i - r_closest_i))

            v_i = self._last_velocities[i]
            v_closest_i = self._last_velocities[closest_idx]

            resides_closest.append(np.heaviside(
                -np.dot(v_i - v_closest_i, r_i - r_closest_i) / initial_distance, 1))
        return resides_closest

    def _check_in_direction(self):
        if self.with_leader:
            return self._check_resides_leader()
        else:
            return self._check_in_direction_without_leader()

    def _check_in_direction_without_leader(self):
        """
        Without leader: checks if agents shift in the specified direction to learn them
        to move in a predefined direction.
        """
        in_direction = []
        for velocity in self._last_velocities:
            vel_abs = np.sqrt(np.dot(velocity, velocity))
            vel_projection = np.dot(velocity, np.array([1/np.sqrt(2), 1/np.sqrt(2)]))
            if vel_projection / vel_abs > 0.99:
                in_direction.append(1)
            else:
                in_direction.append(0)
        return in_direction

    def _check_resides_leader(self):
        """
        With leader: checks if an agent resides within a specified distance to the leader.
        """
        reside_leader = []
        leader_pos = self._last_positions[-1]
        for agent in range(self.n_agents):
            pos_to_leader = self._last_positions[agent] - leader_pos
            current_distance_to_leader = np.sqrt(np.dot(
                pos_to_leader, pos_to_leader))
            reside_leader.append(np.heaviside(
                -current_distance_to_leader + self.distance_to_leader, 1))
        return reside_leader

    def _get_rewards(self, in_comfort_zone, resides_closest,
                     in_direction, dissipations):
        rewards = []
        for i, single_in_comf_zone in enumerate(in_comfort_zone):
            single_resides_closest = resides_closest[i]
            single_dissipation = dissipations[i]
            single_in_direction = in_direction[i]
            reward = self._compute_reward_for_single_agent(
                single_in_comf_zone, single_resides_closest,
                single_in_direction, single_dissipation)
            rewards.append(reward)
        return rewards

    # TODO make kwargs to change attributes for reward counting
    def _compute_reward_for_single_agent(self, in_comfort_zone,
                                         resides_closest,
                                         in_direction,
                                         dissipation):
        """
        Implementation of reward counting algorithm without leader.
        """
        reward = 0
        if in_comfort_zone:
            reward -= 1
        # else:
        if in_direction:
            reward += 2 - max(0, dissipation)
        if resides_closest:
            reward += 1
        return reward

    def _convert_action_to_force(self, actions):
        """
        Converts chosen actions to real force values on the grid.
        Assumes the grid of discrete values of the force.

        Attributes:
            actions (list): contains integer numbers of nodes on the
                            grid of forces.
        """
        grid = self._get_forces_grid()
        forces = []
        for action in actions:
            F_x_idx = action % self.bins_num
            F_y_idx = action // self.bins_num
            F_x = grid[F_x_idx]
            F_y = grid[F_y_idx]
            force_i_mu = np.array([F_x, F_y])
            force_i = np.random.normal(
                loc=force_i_mu, scale=(0.015, 0.015))
            forces.append(force_i)
        return forces

    def _get_forces_grid(self):
        grid = np.linspace(-self._max_force,
                           self._max_force,
                           self.bins_num)
        return grid

    def _transform_state(self, states):
        transformed_states = []
        for state in states:
            transformed_states.append(
                int(state[0] * 4 + state[1] * 2 + state[2]))
        return transformed_states

    def _get_init_state(self):
        closest_agents = self._find_the_closest()
        in_comfort_zone = self._check_closest_in_comfort_zone(closest_agents)
        resides_closest = self._check_resides_closest(closest_agents)
        in_direction = self._check_in_direction()
        init_state = zip(in_comfort_zone, resides_closest, in_direction)
        return init_state

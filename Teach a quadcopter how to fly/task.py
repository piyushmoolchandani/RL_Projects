import numpy as np
from physics_sim import PhysicsSim
import math

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 12
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([20., 33., 10.]) 

    def distance(self, a, b):
        return sum([(a[i] - b[i]) ** 2 for i in range(len(a))])
        #return sum([abs(a[i] - b[i]) for i in range(len(a))]) / 3
    
    def get_reward(self, previous_pose, previous_v, previous_angular_v):
        """Uses current pose of sim to return reward."""
        reward = 0
        if (self.target_pos == self.sim.pose[:3]).all():
            reward += 100
        else:
            a = self.distance(self.target_pos, self.sim.pose[:3])
            b = self.distance(self.target_pos, previous_pose[:3])
            if a >= b:
                reward -= 1
            else:
                reward += 1
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        for i in previous_pose[3:]:
            reward -= 0.05 * math.floor(i)
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        state_all = []
        for _ in range(self.action_repeat):
            previous_pose = self.sim.pose
            previous_v = self.sim.v
            previous_angular_v = self.sim.angular_v
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(previous_pose, previous_v, previous_angular_v) 
            state_all.append(np.concatenate((self.sim.pose, self.sim.v, self.sim.angular_v)))
        next_state = np.concatenate(state_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate((self.sim.pose, self.sim.v, self.sim.angular_v))
        state = np.concatenate([state] * self.action_repeat) 
        return state
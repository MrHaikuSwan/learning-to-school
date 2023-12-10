"""Agent definitions."""
from p5 import *
import numpy as np
import logging
import json


logging.basicConfig(level=logging.DEBUG)


class AgentManager:
    """Manager for all agent data in one simulation."""

    def __init__(self, fish_dna, scale, rng=None, dims=np.array([1000, 1000]), 
                 show_vision_radius=True, edge_mode='wrap', fish_config=None, predator_config=None):
        assert isinstance(fish_dna, FishDna)
        self.agents = []
        self.num_fish = 0
        self.num_predators = 0
        self.num_fish_history = []

        self.fish_dna = fish_dna
        self.scale = scale
        self.dims = dims * scale
        self.fish_rad = 7
        self.predator_rad = 20
        self.show_vision_radius = show_vision_radius
        self.rng = rng if rng is not None else np.random.default_rng(seed=42)
        if edge_mode == 'wrap':
            self.edge_func = self.wrap_edges
        elif edge_mode == 'wall':
            self.edge_func = self.wall_edges

        if fish_config:
            Fish.load_config(fish_config)
        if predator_config:
            Predator.load_config(predator_config)
    
    def wrap_edges(self, pos):
        """Handles simulation borders by wrapping agent position around borders."""
        # NOTE: Should find a better way that extends vision around border
        return np.remainder(pos, self.dims)

    def wall_edges(self, pos):
        """Handles simulation borders by locking agent position within borders."""
        return np.maximum(np.minimum(pos, self.dims), [0, 0])

    def calc_fitness(self):
        """Evaluate fitness of fish as the area under the num_fish_history curve."""
        return sum(self.num_fish_history)

    def populate(self, num_fish, num_predators):
        for _ in range(num_fish):
            self.add_fish()
        for _ in range(num_predators):
            self.add_predator()

    def add_fish(self):
        """Add a randomly initialized Fish to the simulation."""
        pos = self.rng.random(2) * self.dims
        vel = (2 * self.rng.random(2) - 1) * 5
        self.agents.append(Fish(self.get_nearby, pos, vel, self.rng, dna=self.fish_dna))
        self.num_fish += 1

    def add_predator(self):
        """Add a randomly initialize Predator to the simulation."""
        pos = self.rng.random(2) * self.dims
        vel = (2 * self.rng.random(2) - 1) * 5
        self.agents.append(Predator(self.get_nearby, pos, vel, self.rng))
        self.num_predators += 1

    def update(self):
        """Update simulation by one timestep."""
        # Randomize order to avoid any potential unfairness between agents
        random_order = list(range(len(self.agents)))
        self.rng.shuffle(random_order)
        # Update all agent velocities
        for i in random_order:
            self.agents[i].update_vel() # TODO: Hide updated vel from others
        # Updae all agent positions
        for i in random_order:
            self.agents[i].update_pos()
            self.agents[i].pos = self.edge_func(self.agents[i].pos)
        # Kill Fish touching Predators
        for predator in self.agents:
            if isinstance(predator, Predator):
                for fish in self.agents:
                    if isinstance(fish, Fish):
                        between = fish.pos - predator.pos
                        between_threshold = self.fish_rad + self.predator_rad
                        if between @ between < between_threshold ** 2:
                            self.agents.remove(fish)
                            self.num_fish -= 1
        # Update history for fitness
        self.num_fish_history.append(self.num_fish)
    
    def get_nearby(self, target_agent):
        """Get target_agent's nearby agents given their vision radius."""
        vision_rad_sq = target_agent.vision_rad ** 2
        nearby_agents = []
        num_fish = 0
        for agent in self.agents:
            if agent is target_agent:
                continue
            dist_vec = agent.pos - target_agent.pos
            l2_norm_sq = dist_vec @ dist_vec
            if l2_norm_sq < vision_rad_sq:
                agent_info = {
                    'type': type(agent),
                    'pos': agent.pos,
                    'vel': agent.vel,
                    'dist_sq': l2_norm_sq
                }
                if isinstance(agent, Fish):
                    num_fish += 1
                nearby_agents.append(agent_info)
        return nearby_agents, num_fish
    
    def get_scatterplot_data(self):
        colors = [
            'blue' if isinstance(agent, Fish) else 'red'
            for agent in self.agents
        ]
        data = np.array([
            agent.pos
            for agent in self.agents
        ])
        return data, colors
    
    def p5_render(self):
        for agent in self.agents:
            pos = agent.pos / self.scale
            vision_rad = agent.vision_rad / self.scale
            if isinstance(agent, Fish):
                fill('#07c9e3')
                stroke('#07c9e3')
            else:
                fill('#ed7979')
                stroke('#ed7979')
            self._p5_render_triangle(agent)
            if self.show_vision_radius:
                no_fill()
                circle(pos[0], pos[1], int(vision_rad * 2))
    
    def _p5_render_triangle(self, agent):
        pos = agent.pos / self.scale
        agent_rad = self.fish_rad if isinstance(agent, Fish) else self.predator_rad
        coords = agent_rad * 0.1 * np.array([-7, -7, -7, 7, 10, 0]) / self.scale 
        push_matrix()
        translate(pos[0], pos[1])
        rotate((np.pi / 2) - np.arctan2(agent.vel[0], agent.vel[1]))
        triangle(*coords)
        pop_matrix()


class Predator:
    """Predator definition."""
    vision_rad = 300
    min_speed = 2
    max_speed = 10
    max_turn_angle = np.pi / 10
    random_strength = 1

    def __init__(self, f, pos, vel, rng):
        self.pos = pos
        self.vel = vel
        self.alive = True
        self.get_nearby = f
        self.rng = rng

    def update_vel(self):
        """Update Predator velocity with neighbor information abiding by restrictions."""
        nearby_info = self.get_nearby(self)
        vel_update = self.decide_vel_update(nearby_info)
        new_vel = self.vel + vel_update

        # Ensure that new_vel does not allow the predator to turn more than max_turn_rate radians per frame
        cos_alpha = (self.vel @ new_vel) / (np.linalg.norm(self.vel) * np.linalg.norm(new_vel))
        cos_max_turn_angle = np.cos(Predator.max_turn_angle)
        if cos_alpha < cos_max_turn_angle:
            # Get L/R rotation matrices for maximum L/R turn
            # TODO: This is inefficient, should precompute rotation matrices
            L = np.array([
                [np.cos(Predator.max_turn_angle), -np.sin(Predator.max_turn_angle)],
                [np.sin(Predator.max_turn_angle), np.cos(Predator.max_turn_angle)]
            ])
            R = np.array([
                [np.cos(-Predator.max_turn_angle), -np.sin(-Predator.max_turn_angle)],
                [np.sin(-Predator.max_turn_angle), np.cos(-Predator.max_turn_angle)]
            ])
            l_rail = (L @ self.vel) * np.linalg.norm(new_vel) / np.linalg.norm(self.vel)
            r_rail = (R @ self.vel) * np.linalg.norm(new_vel) / np.linalg.norm(self.vel)
            l_sim = l_rail @ new_vel
            r_sim = r_rail @ new_vel
            new_vel = l_rail if l_sim > r_sim else r_rail
        
        # Update self.vel and ensure speed is between min_speed and max_speed
        self.vel = new_vel
        speed = np.linalg.norm(self.vel)
        if speed > Predator.max_speed:
            self.vel *= Predator.max_speed / speed
        elif speed < Predator.min_speed:
            self.vel *= Predator.min_speed / speed

    def update_pos(self):
        """Update Predator position with up-to-date velocity."""
        # TODO: Should change to hide updated velocity from other agents
        self.pos += self.vel

    def decide_vel_update(self, nearby_info):
        """Decide ideal acceleration vector given nearby_info."""
        # Strategy: always accelerate towards the nearest fish as quickly as possible
        nearby_agents, num_fish = nearby_info
        if num_fish == 0:
            return np.array([0, 0], dtype=float)
        nearest_fish = None
        nearest_fish_dist_sq = float('inf')
        for agent_info in nearby_agents:
            if agent_info['type'] == Fish and agent_info['dist_sq'] < nearest_fish_dist_sq:
                nearest_fish_dist_sq = agent_info['dist_sq']
                nearest_fish = agent_info
        assert nearest_fish is not None
        vel_update = nearest_fish['pos'] - self.pos
        random_push = Predator.random_strength * 2 * (self.rng.random(2) - 0.5)
        return vel_update + random_push
    
    @classmethod
    def load_config(cls, fp):
        logging.info("Loading {cls.__name__} config from {fp}")
        with open(fp) as f:
            config_obj = json.load(f)
        cls.vision_rad = config_obj.get('vision_rad', cls.vision_rad)
        cls.min_speed = config_obj.get('min_speed', cls.min_speed)
        cls.max_speed = config_obj.get('max_speed', cls.max_speed)
        cls.random_strength = config_obj.get('random_strength', cls.random_strength)
        cls.max_turn_angle = config_obj.get('max_turn_angle', cls.max_turn_angle)
        logging.info("Loaded config successfully")

    @classmethod
    def write_config(cls, fp):
        logging.info("Writing {cls.__name__} config to {fp}")
        config_obj = {
            'vision_rad': cls.vision_rad,
            'min_speed': cls.min_speed,
            'max_speed': cls.max_speed,
            'random_strength': cls.random_strength,
            'max_turn_angle': cls.max_turn_angle
        }
        with open(fp, 'w') as f:
            json.dump(config_obj, f)
        logging.info("Wrote config successfully")


class Fish:
    """Fish definition."""
    vision_rad = 80
    min_speed = 2
    max_speed = 8
    random_strength = 1

    def __init__(self, f, pos, vel, rng, dna=None):
        self.pos = pos
        self.vel = vel
        self.dna = dna if dna is not None else FishDna()
        self.alive = True
        self.get_nearby = f
        self.rng = rng
    
    def update_vel(self):
        """Update Fish velocity with neighbor information abiding by restrictions."""
        nearby_info = self.get_nearby(self)
        dna_vel_update = self.dna.decide_vel_update(self, nearby_info)
        random_push = Fish.random_strength * 2 * (self.rng.random(2) - 0.5)
        self.vel += dna_vel_update + random_push
        speed = np.linalg.norm(self.vel)
        if speed > Fish.max_speed:
            self.vel *= Fish.max_speed / speed
        elif speed < Fish.min_speed:
            self.vel *= Fish.min_speed / speed

    def update_pos(self):
        """Update Predator position with up-to-date velocity."""
        # TODO: Should change to hide updated velocity from other agents
        self.pos += self.vel
    
    @classmethod
    def load_config(cls, fp):
        logging.info("Loading {cls.__name__} config from {fp}")
        with open(fp) as f:
            config_obj = json.load(f)
        cls.vision_rad = config_obj.get('vision_rad', cls.vision_rad)
        cls.min_speed = config_obj.get('min_speed', cls.min_speed)
        cls.max_speed = config_obj.get('max_speed', cls.max_speed)
        cls.random_strength = config_obj.get('random_strength', cls.random_strength)
        logging.info("Loaded config successfully")
    
    @classmethod
    def write_config(cls, fp):
        logging.info("Writing {cls.__name__} config to {fp}")
        config_obj = {
            'vision_rad': cls.vision_rad,
            'min_speed': cls.min_speed,
            'max_speed': cls.max_speed,
            'random_strength': cls.random_strength
        }
        with open(fp, 'w') as f:
            json.dump(config_obj, f)
        logging.info("Wrote config successfully")


class FishDna:
    """Fish DNA definition. Completely defines a Fish's decisions."""

    def __init__(self, data=None):
        self.data = data if data is not None else {
            'cohesion': np.random.normal(0, 0.5),
            'separation': np.random.normal(0, 0.5),
            'alignment': np.random.normal(0, 0.5),
            'fear': np.random.normal(0, 0.5)
        }
        # self.data = {
        #     'cohesion': 0.05,
        #     'separation': 0.01,
        #     'alignment': 0.05,
        #     'fear': 1,
        # }

    def decide_vel_update(self, target_agent, nearby_info):
        """Decide Fish acceleration vector using DNA parameters given nearby_info."""
        nearby_agents, num_fish = nearby_info
        if not nearby_agents:
            return np.array([0, 0], dtype=float)
        pos = target_agent.pos
        vel = target_agent.vel
        vision_rad = target_agent.vision_rad
        avg_pos = np.array([0, 0], dtype=float)
        avg_vel = np.array([0, 0], dtype=float)
        separation_vec = np.array([0, 0], dtype=float)
        separation_threshold = vision_rad / 2
        fear_vec = np.array([0, 0], dtype=float)
        for agent_info in nearby_agents:
            if agent_info['type'] is Fish:
                # Cohesion + Alignment Averaging
                avg_pos += agent_info['pos']
                avg_vel += agent_info['vel']
                # Separation Nudging
                if agent_info['dist_sq'] < (separation_threshold ** 2):
                    sep_push = agent_info['pos'] - pos
                    if np.any(sep_push):
                        sep_push *= self.data['separation'] / np.linalg.norm(sep_push)
                        separation_vec += sep_push
            else:
                # Fear Nudging
                fear_push = pos - agent_info['pos']
                if np.any(fear_push):
                    fear_push *= self.data['fear'] / np.linalg.norm(fear_push)
                    fear_vec += fear_push
        # Cohesion + Alignment Final Computation
        if num_fish > 0:
            avg_pos /= num_fish
            avg_vel /= num_fish
        cohesion_vec = avg_pos - pos
        if np.any(cohesion_vec):
            cohesion_vec *= self.data['cohesion'] / np.linalg.norm(cohesion_vec)
        alignment_vec = avg_vel - vel
        alignment_vec *= self.data['alignment']
        
        return cohesion_vec + separation_vec + alignment_vec + fear_vec
    
    @classmethod
    def uniform_crossover(cls, dna1, dna2):
        """Create new FishDna by combining two FishDna objects with uniform crossover."""
        crossover_data = {}
        for k in dna1.data:
            if np.random.random() < 0.5:
                crossover_data[k] = dna1.data[k]
            else:
                crossover_data[k] = dna2.data[k]
        return cls(data=crossover_data)
            

                
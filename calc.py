import numpy as np

DEFAULT_PARAMS = {
    "seperation_weight": 1.6,
    "alignment_weight": 1.0,
    "cohesion_weight": 0.9,
    "perception_radius": 55.0,
    "seperation_radius": 22.0,
    "max_speed": 4.0,
    "min_speed": 1.5,
    "max_force": 0.15,
}
TINY_NUM = 1e-9

def normalize(vectors):
    lengths = np.linalg.norm(vectors, axis = 1, keepdims=True)
    return vectors/np.maximum(lengths, TINY_NUM)

def limit(vectors, max_length):
    lengths = np.linalg.norm(vectors, axis = 1, keepdims=True)
    shrink_factor = np.minimum(1.0, max_length/np.maximum(lengths, TINY_NUM))
    return vectors*shrink_factor

class Flock:
    def __init__(self, boid_count, width, height, params = None, see = None):
        self.boid_count = boid_count
        self.world_size = np.array([width, height], dtype = np.float64)
        self.params = params or dict(DEFAULT_PARAMS)
        self.random_generator = np.random.default_rng(see)

        def reset(self):
            self.position = self.random_generator.random((self.boid_count, 2))* self.world_size
            angles = self.random_generator.random(self.boid_count)*2*np.pi
            start_speed = self.params["max_speed"]*0.75
            self.velocities = np.column_stack((np.cos(angles), np.sin(angles)))* start_speed

        def set_boid_count(self, boid_count):
            self.boid_count = boid_count
            self.reset()
            
        def resize(self, width, height):
            self.world_size = np.array([width, height], dtype=np.float64)
            self.positions %= self.world_size

        @property
        def headings(self):
            return np.arctan2(self.velocities[:, 1], self.velocities[:, 0])

        def update(self, time_step = 1.0, extra_accelaration = 0.0):
            accelaration = self._compute_force() + extra_accelaration
            self.velocities += accelaration*time_step
            self._keep_speeds_in_range()
            self.position = (self.positions + self.velocities * time_step)% self.world_size

        def pairwise(self):
            offsets = self.positions[None, :, :] - self.positions[:, None, :]
            offsets -= self.world_size * np.round(offsets/self.world_size)
            squared_distances = np.einsum("ijk, ijk->ij", offsets, offsets)
            return offsets, squared_distances

        def compute_forces(self):
            settings = self.params
            offsets, squared_distances = self.pairwise()
            not_self = ~np.eye(self.boid_count, dtype = bool)

            can_see = (squared_distances < settings["perception_radius"])
        
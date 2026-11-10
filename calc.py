import numpy as np

DEFAULT_PARAMS = {
    "separation_weight": 1.6,
    "alignment_weight": 1.0,
    "cohesion_weight": 0.9,
    "perception_radius": 55.0,
    "separation_radius": 22.0,
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
    def __init__(self, boid_count, width, height, params = None, seed = None):
        self.boid_count = boid_count
        self.world_size = np.array([width, height], dtype = np.float64)
        self.params = params or dict(DEFAULT_PARAMS)
        self.random_generator = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.positions = self.random_generator.random((self.boid_count, 2))* self.world_size
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
        accelaration = self._compute_forces() + extra_accelaration
        self.velocities += accelaration*time_step
        self._keep_speeds_in_range()
        self.positions = (self.positions + self.velocities * time_step)% self.world_size

    def _pairwise(self):
        offsets = self.positions[None, :, :] - self.positions[:, None, :]
        offsets -= self.world_size * np.round(offsets/self.world_size)
        squared_distances = np.einsum("ijk, ijk->ij", offsets, offsets)
        return offsets, squared_distances

    def _compute_forces(self):
        settings = self.params
        offsets, squared_distances = self._pairwise()
        not_self = ~np.eye(self.boid_count, dtype = bool)

        can_see = (squared_distances < settings["perception_radius"]** 2) & not_self
        too_close = (squared_distances < settings["separation_radius"]** 2) & not_self
        can_see = can_see.astype(np.float64)
        too_close = too_close.astype(np.float64)
        neighbour_counts = can_see.sum(axis = 1)

        alignment_force = self._alignment_force(can_see, neighbour_counts)
        cohesion_force = self._cohesion_force(offsets, can_see, neighbour_counts)
        seperation_force = self._separation_force(offsets, squared_distances, too_close)

        return(
            settings["alignment_weight"]* alignment_force
            + settings["cohesion_weight"]*cohesion_force
            +settings["separation_weight"]*seperation_force
        )
    def _alignment_force(self, can_see, neighbour_counts):
        safe_counts = np.maximum(neighbour_counts, 1.0)[:, None]
        average_velocity = (can_see @ self.velocities)/ safe_counts
        return self._steer_toward(average_velocity, neighbour_counts > 0)

    def _cohesion_force(self, offsets, can_see, neighbour_counts):
        safe_counts = np.maximum(neighbour_counts, 1.0)[:, None]
        toward_center = np.einsum("ijk, ij->ik", offsets, can_see) / safe_counts
        return self._steer_toward(toward_center, neighbour_counts > 0)

    def _separation_force(self, offsets, squared_distances, too_close):
        closeness = too_close / np.maximum(squared_distances, 1e-6)
        away_from_crowd = -np.einsum("ijk, ij->ik", offsets, closeness)
        return self._steer_toward(away_from_crowd, too_close.sum(axis = 1)> 0)

    def _steer_toward(self, wanted_direction, has_target):
        settings = self.params
        desired_velocity = normalize(wanted_direction) * settings["max_speed"]
        steering = limit(desired_velocity - self.velocities, settings["max_force"])
        return steering* has_target[:, None]
    
    def _keep_speeds_in_range(self):
        settings = self.params
        speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        clamped_speeds = np.clip(speeds, settings["min_speed"], settings["max_speed"])
        self.velocities *= clamped_speeds / np.maximum(speeds, TINY_NUM)


if __name__ == "__main__":
    import time

    flock = Flock(500, 1280, 720, seed=1)
    flock.update()
    start_time = time.perf_counter()
    frame_count = 100
    for _ in range(frame_count):
        flock.update()

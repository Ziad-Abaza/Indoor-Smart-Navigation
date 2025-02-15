import gym
import numpy as np
from gym import spaces
import pygame
from pygame.locals import QUIT

class SmartCityEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }
    
    def __init__(self):
        super(SmartCityEnv, self).__init__()
        
        self.action_space = spaces.Discrete(4)  
        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.int32),
            high=np.array([20, 20], dtype=np.int32),
            shape=(2,),
            dtype=np.int32
        )
        
        # Initialize road grid
        size = 21
        self.track = np.zeros((size, size), dtype=int)

        # Create horizontal roads (3 rows each)
        self.track[0:3, :] = 1      # North horizontal
        self.track[9:12, :] = 1     # Central horizontal
        self.track[18:21, :] = 1    # South horizontal

        # Create vertical roads (3 columns each)
        self.track[:, 0:3] = 1      # West vertical
        self.track[:, 9:12] = 1     # Central vertical
        self.track[:, 18:21] = 1    # East vertical

        self.tower_points = {
            (10, 10): "A",  # Center
            (18, 18): "B",  # South-east
            (18, 2):  "C",  # South-west
            (2, 18):  "D",  # North-east
            (2, 2):   "E",  # North-west
        }

        self.tile_size = 32
        self.screen_width = size * self.tile_size
        self.screen_height = size * self.tile_size
        self.screen = None
        self.clock = None

        self.agent_position = None
        self.collected_points = set()

        self.reset()

    def reset(self):
        valid_positions = np.argwhere(self.track == 1)
        index = np.random.choice(len(valid_positions))
        self.agent_position = tuple(valid_positions[index])

        self.collected_points = set()
        return np.array(self.agent_position, dtype=np.int32)

    def step(self, action):
        current_r, current_c = self.agent_position
        """
        ######################### table of Actions #############################
        ########################################################################
        # action | name action | (change of row (dr)) | (change of column (dc))#
        #   0    |     top     |         -1           |           0            #
        #   1    |   bottom    |         1            |           0            #
        #   2    |   left      |         0            |          -1            #
        #   3    |   right     |         0            |           1            #
        #  none  |   no move   |         0            |           0            #
        ########################################################################
        """
        # Determine allowed actions based on road type
        in_horizontal = current_r in {0,1,2,9,10,11,18,19,20}
        in_vertical = current_c in {0,1,2,9,10,11,18,19,20}
        
        if in_horizontal and not in_vertical:
            allowed = [2, 3]  # Left/Right
        elif in_vertical and not in_horizontal:
            allowed = [0, 1]  # Up/Down
        else:
            allowed = [0, 1, 2, 3]  # Intersection
            
        if action not in allowed:
            dr, dc = 0, 0
        else:
            if action == 0:    dr, dc = -1, 0
            elif action == 1:  dr, dc = 1, 0
            elif action == 2:  dr, dc = 0, -1
            elif action == 3:  dr, dc = 0, 1
            else:              dr, dc = 0, 0

        new_r = current_r + dr
        new_c = current_c + dc

        if 0 <= new_r < 21 and 0 <= new_c < 21 and self.track[new_r, new_c] == 1:
            self.agent_position = (new_r, new_c)

        # Reward calculation
        reward = 0.0
        if self.agent_position in self.tower_points and self.agent_position not in self.collected_points:
            reward += 1.0
            self.collected_points.add(self.agent_position)

        done = len(self.collected_points) == len(self.tower_points)
        return np.array(self.agent_position, dtype=np.int32), reward, done, {}

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()

        # Define colors
        COLOR_BLOCK = (50, 50, 50)
        COLOR_ROAD = (100, 100, 100)
        COLOR_DIVIDER = (255, 255, 0)
        COLOR_AGENT = (255, 0, 0)
        COLOR_POINT = (160, 32, 240)
        COLOR_COLLECTED = (220, 160, 220)

        # Draw grid
        self.screen.fill((0, 0, 0))
        for r in range(21):
            for c in range(21):
                x = c * self.tile_size
                y = r * self.tile_size
                
                if self.track[r, c] == 1:
                    # Draw road
                    pygame.draw.rect(self.screen, COLOR_ROAD, (x, y, self.tile_size, self.tile_size))
                    
                    # Draw dividers
                    if r in [1,10,19] and c not in [0,1,2,9,10,11,18,19,20]:
                        pygame.draw.line(self.screen, COLOR_DIVIDER, 
                                       (x, y+self.tile_size//2),
                                       (x+self.tile_size, y+self.tile_size//2), 3)
                        
                    if c in [1,10,19] and r not in [0,1,2,9,10,11,18,19,20]:
                        pygame.draw.line(self.screen, COLOR_DIVIDER,
                                       (x+self.tile_size//2, y),
                                       (x+self.tile_size//2, y+self.tile_size), 3)
                
                # Draw points
                if (r,c) in self.tower_points:
                    color = COLOR_COLLECTED if (r,c) in self.collected_points else COLOR_POINT
                    pygame.draw.circle(self.screen, color, 
                                    (x+self.tile_size//2, y+self.tile_size//2),
                                    self.tile_size//3)

        # Draw agent
        ax = self.agent_position[1] * self.tile_size
        ay = self.agent_position[0] * self.tile_size
        pygame.draw.rect(self.screen, COLOR_AGENT, (ax, ay, self.tile_size, self.tile_size))

        pygame.display.update()
        self.clock.tick(10)

        if mode == 'rgb_array':
            return pygame.surfarray.array3d(self.screen)
        return None

    def close(self):
        pygame.quit()

# Test the environment
if __name__ == "__main__":
    env = SmartCityEnv()
    obs = env.reset()
    done = False
    
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        env.render()
    
    print("Total collected points:", len(env.collected_points))
    env.close()
import numpy as np
import os
import json


if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    

class MDP:
    """
    mat : np.array, tensor consisting of the current state and postgrid as a 4X4 matrix
    agentPosition : int tuple, the current position of the agent in the current state
    gamma : discount factor
    """
    gamma = 0.5

    def __init__(self, dir = "data", type = "train", name = "0", lambda1 = 0.01, lambda2 = 0.1, lambda3 = 1):
        """
        dir : string
            data, data_easy or data_medium : the directory the task is located in
        type : string
            train, val or test 
        name : str
            the number of the gridworld task as in the /datasets directory
        """
        path = os.sep.join(["datasets", dir, type, "task", name + "_task.json"])
        self.parse_json(path)
        self.lambda1, self.lambda2, self.lambda3 = lambda1, lambda2, lambda3
        self.lastManDist = self.sum_of_goals()

    ##################################
    #   Helper Functions             #
    ##################################
    def task_solved(self):
        return np.array_equal(*self.matrix)

    def get_current_state(self):
        return self.matrix


    def print_grid(self):
        symbols = {0 : ".", 
                    1: "<",
                    2: "v",
                    3: ">",
                    4: "^",
                    5: "l",
                    6: "d",
                    7: "r",
                    8: "u",
                    9: "O",
                    10: "#"}

        print_mat = np.chararray(shape = self.matrix.shape)
        print_mat = np.vectorize(symbols.get)(self.matrix)
        print(print_mat)


    def parse_json(self, path):
        """
        parses the json input and generates the pre- and post-grid matrix
        """
        with open(path) as file:
            grid = json.load(file)
            # create matrix 
            mat = np.zeros(shape= (2, grid["gridsz_num_rows"], grid["gridsz_num_cols"]))
            
            # add walls
            if grid["walls"]:
                mat[0][tuple(np.array(grid["walls"]).T)] = 10
                mat[1][tuple(np.array(grid["walls"]).T)] = 10
            
            # add agent
            directions = {"west" : 1, "south" : 2, "east" : 3, "north" : 4}
            self.agentPosition = grid["pregrid_agent_row"], grid["pregrid_agent_col"]
            mat[0][self.agentPosition] = directions[grid["pregrid_agent_dir"]]
            mat[1][grid["postgrid_agent_row"], grid["postgrid_agent_col"]] = directions[grid["postgrid_agent_dir"]]

            # add Markers to pregrid
            for marker in grid["pregrid_markers"]:
                if mat[(0, *marker)] == 0:
                    mat[(0, *marker)] = 9
                else:
                    mat[(0, *marker)] += 4

            #add Markers to pregrid
            for marker in grid["postgrid_markers"]:
                if mat[(1, *marker)] == 0:
                    mat[(1, *marker)] = 9
                else:
                    mat[(1, *marker)]

            self.matrix = mat
    

    def marker_on_pos(self, pos):
        return self.matrix[pos] in set(range(5,10))

    
    def get_change_in_direction(self):
        return {
            1 : (0,-1),
            2 : (1,0),
            3 : (0,1),
            4 : (-1,0),
            5 : (0,-1),
            6 : (1,0),
            7 : (0,1),
            8 : (-1,0)
        }.get(self.matrix[(0,*self.agentPosition)])

    def hit_wall(self, pos):
        return self.matrix[pos] == 10

    def out_of_bounds(self, pos):
        x, y = pos[1:3]
        return x < 0 or x >= self.matrix.shape[1] or y < 0 or y >= self.matrix.shape[2]

    def sum_of_goals(self):
        # compare post and pre grid except on the agents positions for change, as these are the goals
        change = self.matrix[0] != self.matrix[1]
        change[self.agentPosition] = False
        # get matrix of indices
        indxs = np.indices(change.shape)
        indxs = np.dstack((*indxs,))

        #get manhattan distance of every coordinate
        manDist = indxs - self.agentPosition
        manDist = np.sum(np.abs(manDist), axis = 2) * change

        return np.sum(manDist)
        

    ################################################
    #   Reward Function                            #
    ################################################

    def reward(self, action):
        if action in {"turnRight", "turnLeft"}:
            return 0

        if action == "move":
            newManDist = self.sum_of_goals()
            if newManDist > self.lastManDist:
                i = -1
            else:
                i = 1
            self.lastManDist = newManDist
            return i * self.lambda1

        if action == "pickMarker":
            # marker on agents position
            if self.marker_on_pos((0,*self.agentPosition)) and (not self.marker_on_pos((1,*self.agentPosition))):
                return self.lambda2
            else:
                return -self.lambda2

        if action == "putMarker":
            if (not self.marker_on_pos((0,*self.agentPosition))) and self.marker_on_pos((1, *self.agentPosition)):
                return self.lambda2
            else:
                return -self.lambda2

        if action == "finish":
            if np.array_equal(self.matrix[0], self.matrix[1]):
                return self.lambda3
            else:
                return -self.lambda3

    #################################################
    #   Transition Dynamics                         #
    #################################################
    
    def get_next_state(self, action):
        """
        NOTE: Do not call this function to get the next state, use the reward function instead, as this function could cause bugs in the reward if called alone, don't let it be lonely :(
        """
        AP = (0, *self.agentPosition)
        if action == "move":

            # save S_curr_Ap 
            newij = self.matrix[(0,*self.agentPosition)]
            newAP = 0

            ij = (0,*(np.array(self.agentPosition) + np.array(self.get_change_in_direction())))

            #check crash
            if self.out_of_bounds(ij) or self.hit_wall(ij):
                return "Terminal"

            # check markers
            if self.marker_on_pos(AP):
                newAP = 9
                if not self.marker_on_pos(ij):
                    newij -= 4
            else:
                if self.marker_on_pos(ij):
                    newij += 4

            # change current position
            self.matrix[AP] = newAP
            
            # move agent to next position
            self.matrix[ij] = newij 

            self.agentPosition = ij[1:3]

        if action == "turnLeft":
            self.matrix[AP] = (self.matrix[AP]) % 4 + 1 + self.marker_on_pos(AP) * 4

        if action == "turnRight":
            self.matrix[AP] = (self.matrix[AP] -2 ) % 4 + 1 + self.marker_on_pos(AP) * 4

        if action == "pickMarker":
            if self.marker_on_pos(AP):
                self.matrix[AP] -= 4
            else:
                return "Terminal"

        if action == "putMarker":
            if self.marker_on_pos(AP):
                return "Terminal"
            else:
                self.matrix[AP] += 4

        if action == "finish":
            return "Terminal"


    def sample_next_state_and_reward(self, action):
        # when move is action, the next state is important, else, the current one is
        if action == "move":
            done = self.get_next_state(action) == "Terminal"
            return self.get_current_state(), self.reward(action),  done, {}

        else:
            rew = self.reward(action)
            done = self.get_next_state(action) == "Terminal"
            return self.get_current_state(), rew, done, {}
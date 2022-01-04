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

    def __init__(self, dir = "data", type = "train", name = "0"):
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

    ##################################
    #   Helper Functions             #
    ##################################
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
            print(grid)
            # create matrix 
            mat = np.zeros(shape= (2, grid["gridsz_num_rows"], grid["gridsz_num_cols"]))
            
            # add walls
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
            self.print_grid()
    

    def marker_on_pos(self, pos):
        return self.matrix[pos] in set(range(5,9))

    
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
        return x < 0 or x > self.matrix.shape[1] or y < 0 or y > self.matrix.shape[2]

    ################################################
    #   Reward Function                            #
    ################################################

    def reward(self, action):
        if action in {"move", "turnRight", "turnLeft"}:
            return 0

        if action == "pickMarker":
            # marker on agents position
            if self.marker_on_pos((0,*self.agentPosition)) and not self.marker_on_pos((1,*self.agentPosition)):
                return 1
            else:
                return -1

        if action == "putMarker":
            if not self.marker_on_pos((0,*self.agentPosition)) and self.marker_on_pos((1, *self.agentPosition)):
                return 1
            else:
                return -1

        if action == "finish":
            if np.array_equal(self.matrix[0], self.matrix[1]):
                return 10
            else:
                return -10

    #################################################
    #   Transition Dynamics                         #
    #################################################
    
    def get_next_state(self, action):
        AP = (0, *self.agentPosition)
        if action == "move":

            # save S_curr_Ap 
            newij = self.matrix[(0,*self.agentPosition)]
            newAP = 0

            ij = (0,*(np.array(self.agentPosition) + np.array(self.get_change_in_direction())))

            #check crash
            if self.hit_wall(ij) or self.out_of_bounds(ij):
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


test = MDP("data_easy", name = "0")
print(test.get_next_state("putMarker"))
test.get_next_state("pickMarker")
test.print_grid()
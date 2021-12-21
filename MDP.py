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

        
MDP("data_easy", name = "4000")
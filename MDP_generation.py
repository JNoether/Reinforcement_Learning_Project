from MDP import MDP
import os
import sys
import numpy as np
import json
import re
import random

def out_of_bounds(agent_pos):
    x,y = agent_pos
    return x < 0 or x >= 4 or y < 0 or y >= 4

if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    #number of MDPs to be generated
    # float to allow 1e6
    num_generations = int(float(sys.argv[1]))
    #probability that wall is placed
    mode = sys.argv[2]

    # used for file name
    try:
        max_file = max([int(re.sub(r"\D", "", i)) for i in os.listdir(f"datasets/generated_{mode}/train/task")])
    except:
        max_file = 0
    for i in range(1,num_generations+1):
        #print progress
        if i % (num_generations/10) == 0:
            print(f"{i*100/ num_generations} %")

        #json
        grid = {"gridsz_num_rows": 4, "gridsz_num_cols": 4}

        # agent post-grid position
        directions = ["east", "south", "west", "north"]
        post_agent_x, post_agent_y, post_agent_dir = np.random.randint(0,3), np.random.randint(0,3), np.random.randint(0,3)

        grid["postgrid_agent_row"] = int(post_agent_x)
        grid["postgrid_agent_col"] = int(post_agent_y)
        grid["postgrid_agent_dir"] = directions[post_agent_dir]


        #generate wall positions:
        walls = set()
        wall_prob = {"easy" : 0.6, "med" : 0.75, "hard": 0.85}.get(mode)
        while np.random.rand() < wall_prob and len(walls) < 12:
            wall_pos = random.randint(0, 3), random.randint(0, 3)
            if wall_pos != (post_agent_x, post_agent_y):
                walls.add(wall_pos)
        
        grid["walls"] = [list(x) for x in walls]

        assert len(walls) < 16

        
        #generate pregrid and postgrid markers
        markers = set()
        marker_prob = {"easy" : 0.05, "med" : 0.1, "hard" : 0.15}.get(mode)
        max_markers = {"easy": 1, "med": 2, "hard" : 3}.get(mode)
        while len(markers) < max_markers and np.random.rand() < marker_prob:
            x, y = random.randint(0, 3), random.randint(0, 3)
            if (x,y) not in walls:
                markers.add((x,y))


        #generate (solvable) pregrid by applying actions backward
        changeInDirection = {0 : np.array([0,1]), 1 : np.array([-1,0]), 2 : np.array([0,-1]), 3 : np.array([1,0])}

        agentPos = np.array([post_agent_x, post_agent_y])
        preMarkers = markers.copy()
        postMarkers = markers.copy()
        lengths= {"easy" : [1,2,3,4], "med" : [2, 3, 4, 5], "hard" : [4, 5, 6, 7]}.get(mode)
        episodes = np.random.choice(lengths, p = [0.2, 0.3, 0.3, 0.2])
        j = 0
        while j < episodes:
            j += 1
            newPos = agentPos - changeInDirection[post_agent_dir]
            newPos = (int(newPos[0]), int(newPos[1]))
            if newPos in walls or out_of_bounds(newPos):
                action = np.random.choice(["turnLeft", "turnright"])
            else:
                action = np.random.choice(["move", "turnLeft", "turnright"], p = [0.6, 0.2, 0.2])
            if action == "move":
                agentPos -= changeInDirection[post_agent_dir]
            if action == "turnLeft":
                post_agent_dir = (post_agent_dir + 1) % 4
            if action == "turnRight":
                post_agent_dir = (post_agent_dir - 1) % 4

            #randomly add pick/put marker:
            #pick
            if np.random.rand() < marker_prob:
                j += 1
                preMarkers.add((int(agentPos[0]), int(agentPos[1])))
            #put
            if np.random.rand() < marker_prob:
                j += 1
                markers.add((int(agentPos[0]), int(agentPos[1])))
            
        grid["pregrid_agent_row"] = int(agentPos[0])
        grid["pregrid_agent_col"] = int(agentPos[1])
        grid["pregrid_agent_dir"] = directions[post_agent_dir]

        grid["pregrid_markers"] = [list(x) for x in preMarkers]
        grid["postgrid_markers"] = [list(x) for x in markers]
        
        with open(f"datasets/generated_{mode}/train/task/{max_file + i}_task.json", "w+") as file:
            json.dump(grid, file)
        
    print("done")

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
    wall_prob = float(sys.argv[2])
    #probability for all markers(placed/picked up)
    marker_prob = float(sys.argv[3])

    assert wall_prob > 0 and wall_prob < 1

    # used for file name
    max_file = max([int(re.sub(r"\D", "", i)) for i in os.listdir("datasets/generated/train/task")])
    for i in range(1,num_generations+1):
        #print progress
        if i % (num_generations/10) == 0:
            print(f"{i*100/ num_generations} %")

        #json
        grid = {"gridsz_num_rows": 4, "gridsz_num_cols": 4}

        #generate wall positions:
        walls = set()
        while np.random.rand() < wall_prob and len(walls) < 12:
            wall_pos = random.randint(0, 3), random.randint(0, 3)
            walls.add(wall_pos)
        grid["walls"] = [list(x) for x in walls]

        assert len(walls) < 16

        # agent post-grid position
        directions = ["east", "south", "west", "north"]
        while True:
            post_agent_x, post_agent_y, post_agent_dir = np.random.randint(0,3), np.random.randint(0,3), np.random.randint(0,3)
            if (post_agent_x, post_agent_y) not in walls:
                break

        grid["postgrid_agent_row"] = int(post_agent_x)
        grid["postgrid_agent_col"] = int(post_agent_y)
        grid["postgrid_agent_dir"] = directions[post_agent_dir]

        #generate pregrid and postgrid markers
        markers = set()
        while len(markers) < 3 and np.random.rand() < marker_prob:
            x, y = random.randint(0, 3), random.randint(0, 3)
            if (x,y) not in walls:
                markers.add((x,y))


        #generate (solvable) pregrid by applying actions backward
        changeInDirection = {0 : np.array([0,1]), 1 : np.array([-1,0]), 2 : np.array([0,-1]), 3 : np.array([1,0])}

        agentPos = np.array([post_agent_x, post_agent_y])
        preMarkers = markers.copy()
        episodes = np.random.choice([2,3,4,5,6,7], p = [0.15, 0.2, 0.2, 0.3, 0.1, 0.05])
        for j in range(episodes):
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

        with open(f"datasets/generated/train/task/{max_file + i}_task.json", "w+") as file:
            json.dump(grid, file)
        
    print("done")

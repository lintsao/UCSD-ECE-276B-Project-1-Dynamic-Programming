# from utils import *
# from example import example_use_of_gym_env

# MF = 0  # Move Forward
# TL = 1  # Turn Left
# TR = 2  # Turn Right
# PK = 3  # Pickup Key
# UD = 4  # Unlock Door


# def doorkey_problem(env):
#     """
#     You are required to find the optimal path in
#         doorkey-5x5-normal.env
#         doorkey-6x6-normal.env
#         doorkey-8x8-normal.env

#         doorkey-6x6-direct.env
#         doorkey-8x8-direct.env

#         doorkey-6x6-shortcut.env
#         doorkey-8x8-shortcut.env

#     Feel Free to modify this fuction
#     """
#     optim_act_seq = [TL, MF, PK, TL, UD, MF, MF, MF, MF, TR, MF]
#     return optim_act_seq


# def partA():
#     env_path = "./envs/known_envs/doorkey-8x8-normal.env"
#     env, info = load_env(env_path)  # load an environment
#     seq = doorkey_problem(env)  # find the optimal action sequence
#     draw_gif_from_seq(seq, load_env(env_path)[0])  # draw a GIF & save


# def partB():
#     env_folder = "./envs/random_envs"
#     env, info, env_path = load_random_env(env_folder)


# if __name__ == "__main__":
#     # example_use_of_gym_env()
#     partA()
#     # partB()


#In[]
import numpy as np
import gym
import math
import gym_minigrid
from utils import *
import matplotlib.pyplot as plt
# %%
MF = 0 # Move Forward
TL = 1 # Turn Left
TR = 2 # Turn Right
PK = 3 # Pickup Key
UD = 4 # Unlock Door

Action=np.array([MF,TL,TR,PK,UD])
#In[]
# When key is needed
def robot_motion(grid,env): # Function to make the robot move when direct path not available
    l= env.agent_pos[1]
    b= env.agent_pos[0]

    right_grid_F=grid[l,b+1]
    left_grid_F=grid[l,b-1]
    top_grid_F=grid[l-1,b]
    bottom_grid_F=grid[l+1,b]

    a= (np.where(np.array([right_grid_F,left_grid_F,top_grid_F,bottom_grid_F]) < grid[l,b]))[0][0]
    dir=env.agent_dir
    print(a)
    print(dir)
    if(dir==0):
        if(a==0):
            print('MF')
            step(env,MF)
            return [MF]
        elif(a==1):
            print('TR -> TR -> MF')
            step(env,TR)
            step(env,TR)
            step(env,MF)
            return [TR,TR,MF]
        elif(a==2):
            print('TL-> MF')
            step(env,TL)
            step(env,MF)
            return [TL,MF]
        elif(a==3):
            print('TR -> MF')
            step(env,TR)
            step(env,MF)
            return [TR,MF]
        # theta= np.pi/2 # 90
    elif(dir==1):
        if(a==0):
            print('TL -> MF')
            step(env,TL)
            step(env,MF)
            return [TL,MF]
        elif(a==1):
            print('TR -> MF')
            step(env,TR)
            step(env,MF)
            return [TR,MF]
        elif(a==2):
            print('TR -> TR -> MF')
            step(env,TR)
            step(env,TR)
            step(env,MF)
            return [TR,TR,MF]
        elif(a==3):
            print('MF')
            step(env,MF)
            return [MF]
        # theta=0
    elif(dir==2):
        if(a==0):
            print('TR -> TR -> MF')
            step(env,TR)
            step(env,TR)
            step(env,MF)
            return [TR,TR,MF]
        elif(a==1):
            print('MF')
            step(env,MF)
            return [MF]
        elif(a==2):
            print('TR-> MF')
            step(env,TR)
            step(env,MF)
            return [TR,MF]
        elif(a==3):
            print('TL -> MF')
            step(env,TL)
            step(env,MF)
            return [TL,MF]
        # theta= -np.pi/2 #-90
    elif(dir==3):
        if(a==0):
            print('TR -> MF')
            step(env,TR)
            step(env,MF)
            return [TR,MF]
        elif(a==1):
            print('TL -> MF')
            step(env,TL)
            step(env,MF)
            return [TL,MF]
        elif(a==2):
            print('MF')
            step(env,MF)
            return [MF] 
        elif(a==3):
            print('TR -> TR -> MF')
            step(env,TR)
            step(env,TR)
            step(env,MF)
            return [TR,TR,MF]
        # theta=np.pi


#In[]
# When key is not needed
def robot_2_Grid(grid,env): # Function to make the robot move when direct path is available
    l= env.agent_pos[1]
    b= env.agent_pos[0]

    right_grid_F=grid[l,b+1]
    left_grid_F=grid[l,b-1]
    top_grid_F=grid[l-1,b]
    bottom_grid_F=grid[l+1,b]

    a= np.where(np.array([right_grid_F,left_grid_F,top_grid_F,bottom_grid_F]) < grid[l,b])[0] 
    dir=env.agent_dir
    print(a)
    print(dir)
    if(dir==0):
        if(a==0):
            print('MF')
            step(env,MF)
            return [MF]
        elif(a==1):
            print('TR -> TR -> MF')
            step(env,TR)
            step(env,TR)
            step(env,MF)
            return [TR,TR,MF]
        elif(a==2):
            print('TL-> MF')
            step(env,TL)
            step(env,MF)
            return [TL,MF]
        elif(a==3):
            print('TR -> MF')
            step(env,TR)
            step(env,MF)
            return [TR,MF]
        # theta= np.pi/2 # 90
    elif(dir==1):
        if(a==0):
            print('TL -> MF')
            step(env,TL)
            step(env,MF)
            return [TL,MF]
        elif(a==1):
            print('TR -> MF')
            step(env,TR)
            step(env,MF)
            return [TR,MF]
        elif(a==2):
            print('TR -> TR -> MF')
            step(env,TR)
            step(env,TR)
            step(env,MF)
            return [TR,TR,MF]
        elif(a==3):
            print('MF')
            step(env,MF)
            return [MF]
        # theta=0
    elif(dir==2):
        if(a==0):
            print('TR -> TR -> MF')
            step(env,TR)
            step(env,TR)
            step(env,MF)
            return [TR,TR,MF]
        elif(a==1):
            print('MF')
            step(env,MF)
            return [MF]
        elif(a==2):
            print('TR-> MF')
            step(env,TR)
            step(env,MF)
            return [TR,MF]
        elif(a==3):
            print('TL -> MF')
            step(env,TL)
            step(env,MF)
            return [TL,MF]
        # theta= -np.pi/2 #-90
    elif(dir==3):
        if(a==0):
            print('TR -> MF')
            step(env,TR)
            step(env,MF)
            return [TR,MF]
        elif(a==1):
            print('TL -> MF')
            step(env,TL)
            step(env,MF)
            return [TL,MF]
        elif(a==2):
            print('MF')
            step(env,MF)
            return [MF] 
        elif(a==3):
            print('TR -> TR -> MF')
            step(env,TR)
            step(env,TR)
            step(env,MF)
            return [TR,TR,MF]
        # theta=np.pi


# In[]
def doorkey_problem(flag,c_CD,c_OD_1,c_OD_2,c_OD_3,goal,agentPos,keyPos,doorPos,env,info):
    '''
    Fuction to build the sequence of control actions. 
    This funtion calls the robot_2Grid and robot_motion function
    Flag=True => We need key to reach goal
    Flag=False => We can reach goal directly
    '''
    goal=np.roll(goal,1)
    agentPos=np.roll(agentPos,1)
    keyPos=np.roll(keyPos,1)
    doorPos=np.roll(doorPos,1)

    if(flag==True):
        # print(c_CD)
        print('Key needed') # work with other 3 matrices here

        count1=c_OD_1[agentPos[0],agentPos[1]]-1
        print('count',count1)
        print('cost_grid',c_OD_1)

        seq=[]
        # plot_env(env)
        while count1>=0: # While loop to go from Initial Positon to Key position
            val=(robot_motion(c_OD_1,env))
            if(val):    
                for i in val:
                    seq.append(i)
            # elif(not val):
            #     seq.append(MF)
            # plot_env(env)
            print('count1',count1)
            count1=count1-1
        
        print(seq)
        seq.pop(-1)
        print(seq)
        # seq.pop(-1)
        seq.append(PK)
        step(env,PK) # Pick up the key
        print(seq)

        agentPos=env.agent_pos
        agentPos=np.roll(agentPos,1)

        # count2=c_OD_2[keyPos[0],keyPos[1]]-1
        count2=c_OD_2[agentPos[0],agentPos[1]]-1
        print(count2)
        print(c_OD_2)

        while count2>=0: # While loop to go from key Positon to Door position
            val=(robot_motion(c_OD_2,env))
            if(val):    
                for i in val:
                    seq.append(i)
            # elif(not val):
            #     seq.append(MF)
            # plot_env(env)
            print('count2',count2)
            count2=count2-1
        
        # seq.pop(-1)
        print(seq)
        seq.pop(-1)
        seq.append(UD) # Unlock the Door
        step(env,UD)
        print(seq)

        agentPos=env.agent_pos
        agentPos=np.roll(agentPos,1)

        # count3=c_OD_3[doorPos[0],doorPos[1]]
        count3=c_OD_3[agentPos[0],agentPos[1]]-1
        print(count3)
        print(c_OD_3)
        
        while count3>=0: # While loop to go from Door Positon to Goal position
            val=(robot_motion(c_OD_3,env))
            if(val):    
                for i in val:
                    seq.append(i)
            # elif(not val):""
            #     seq.append(MF)
            # plot_env(env)
            print('count3',count3)
            count3=count3-1
        # seq.pop(-1)
        
        optim_act_seq=seq
    else: # When Direct path available
        print('Key not needed')
        count=c_CD[agentPos[0],agentPos[1]]
        print(count)
        seq=[]
        # plot_env(env)
        while count>=0: # While loop to go from Initial Positon to Door position Directly
            val=(robot_2_Grid(c_CD,env))
            if(val):    
                for i in val:
                    seq.append(i)
            # elif(not val):
            #     seq.append(MF)
            # plot_env(env)
            print(count)
            count=count-1
        print(c_CD)
        optim_act_seq=seq

        # print()
        
    # optim_act_seq=seq
    # optim_act_seq = [TL, MF, PK, TL, UD, MF, MF, MF, MF, TR, MF]
    return optim_act_seq

#In[]
def fill_4_cells(cost_grid,loc,val,grid_flag): 
    '''
    Fill the 4 cells surrounding 
    a given cell (whose value is K) 
    with K+1(if the cell is not visited earlier) 
    else leave the value as it is
    '''
    l=loc[0]
    b=loc[1]
    # print(l,b)
    # r=cost_grid[l,b]+1
    if cost_grid[l-1,b]!= math.inf and grid_flag[l-1,b]==0:
        cost_grid[l-1,b]=cost_grid[l,b]+1 #Top
        grid_flag[l-1,b]=1
    else:
    #     cost_grid[l-1,b]=math.inf
        grid_flag[l-1,b]=1

    if cost_grid[l+1,b]!= math.inf and grid_flag[l+1,b]==0 :
        cost_grid[l+1,b]=cost_grid[l,b]+1 #Bottom
        grid_flag[l+1,b]=1
    else:
    #     cost_grid[l+1,b]=math.inf
        grid_flag[l+1,b]=1

    if cost_grid[l,b-1]!= math.inf and grid_flag[l,b-1]==0:
        cost_grid[l,b-1]=cost_grid[l,b]+1 #Left
        grid_flag[l,b-1]=1
    else:
    #     cost_grid[l,b-1]=math.inf
        grid_flag[l,b-1]=1

    if cost_grid[l,b+1]!= math.inf and grid_flag[l,b+1]==0:
        cost_grid[l,b+1]=cost_grid[l,b]+1 #Right
        grid_flag[l,b+1]=1
    else:
    #     cost_grid[l,b+1]=math.inf
        grid_flag[l,b+1]=1

    # a=min(np.max(cost_grid),cost_grid[l,b]+1)
    # print('a------------------------>>> ',a)
    return cost_grid,grid_flag,(cost_grid[l,b]+1)

#In[]
def label_Correction(env,agentPos,cost_grid,goal,grid_flag):
    '''
    This function employes label correction algo with
    the help of fill_4_cells fuction above.
    '''

    c=0
    goal=np.roll(goal,1)
    agentPos=np.roll(agentPos,1)
    cost_grid,grid_flag,r = fill_4_cells(cost_grid, goal,0, grid_flag)
    # print('here')
    # print('Cost_grid')
    # print(cost_grid)
    # print('Grid_flag')
    # print(grid_flag)
    
    while c<=r:
        
        q=np.vstack((np.where(cost_grid==r)[0],np.where(cost_grid==r)[1]))
        # print(q)
        grid_flag[goal[0],goal[1]]=1
        # print(y)
        for i in range(len(q.T)):
            cost_grid,grid_flag,r= fill_4_cells(cost_grid, q[:,i].T,0, grid_flag)
        c=c+1

    # print('Cost_grid')
    # print(cost_grid)
    # print('Grid_flag')
    # print(grid_flag)
#In[]
# def get_values(l,b):
#     return  np.array([cost_grid[l+1,b], cost_grid[l-1,b], cost_grid[l,b+1], cost_grid[l,b-1])

#In[]
def plot_value_function(env,seq,goal,agentPos,doorPos, flag,info):
    '''
    Fuction to plot the Value function of each cell. As the agent follows the shortest path
    '''

    print('inside value function')
    l=goal[1]
    b=goal[0]

    l_keypos=info['key_pos'][1]
    b_keypos=info['key_pos'][0]
    
    l_doorPos=info['door_pos'][1]
    b_doorPos=info['door_pos'][0]

    print(l_doorPos,b_doorPos)
    print(l,b)
    print(l_keypos,b_keypos)

    Q=np.zeros((9,len(seq)))
    # seq.append(MF)
    if(UD in seq):
        store =seq.index(UD)
    else:
        store=1

    for i in range(len(seq)):
        
        # goal=np.roll(env.agent_pos,1)
        goal=env.agent_pos
        # plot_env(env)
        print('===========================')
        world_grid= (gym_minigrid.minigrid.Grid.encode(env.grid)[:,:,0].T).astype(np.float32)
        # world_grid[np.where(world_grid==2)]=math.inf
        index= np.where(world_grid!=1 )
        world_grid[index[0][:],index[1][:]]= math.inf
        world_grid[info['key_pos'][1],info['key_pos'][0]]=-2
        world_grid[info['goal_pos'][1],info['goal_pos'][0]]=0

        world_grid[np.where(world_grid==1)]=0
        grid_flag=np.zeros(np.shape(world_grid))
        
        cost_grid=world_grid
        cost_grid[np.where(cost_grid<=0)]=0   
        grid_flag=np.zeros(np.shape(world_grid))     
        
        cost_grid[l+1,b]=math.inf
        cost_grid[l-1,b]=0
        cost_grid[l,b+1]=math.inf
        cost_grid[l,b-1]=0

        print("print wala", seq[i])

        if flag==True:
            Q[4,0:store]=15 #cost_grid[l+1,b]
            Q[5,0:store]=15 #cost_grid[l-1,b]
            Q[6,0:store]=15 #cost_grid[l,b+1]
            Q[7,0:store]=15 #cost_grid[l,b-1]

            cost_grid[doorPos[1],doorPos[0]]=0
            world_grid[info['door_pos'][1]][info['door_pos'][0]]=0
            print(cost_grid)
            print('GOAL:  ',goal)
            label_Correction(env,agentPos,cost_grid,goal,grid_flag)
            # if(env.)
            print(grid_flag)
            print(cost_grid)
            step(env,seq[i])
            if(seq[i]==UD):
                store=i
            
            Q[0,i]=cost_grid[l_keypos+1,b_keypos]
            Q[1,i]=cost_grid[l_keypos-1,b_keypos]
            Q[2,i]=cost_grid[l_keypos,b_keypos+1]
            Q[3,i]=cost_grid[l_keypos,b_keypos-1]
            
            Q[4,i]=cost_grid[l+1,b]
            Q[5,i]=cost_grid[l-1,b]
            Q[6,i]=cost_grid[l,b+1]
            Q[7,i]=cost_grid[l,b-1]
            
            # Q[8,i]=cost_grid[l_doorPos,b_doorPos+1]
            Q[8,i]=cost_grid[l_doorPos,b_doorPos-1]

            

            # plot_env(env)
            #### STORE Values#########
            
            print('-------------------------------------------------------------')
        
        else:     # We have shortcut
            print(cost_grid)
            Q[4,0:store]=15 #cost_grid[l+1,b]
            Q[5,0:store]=15 #cost_grid[l-1,b]
            Q[6,0:store]=15 #cost_grid[l,b+1]
            Q[7,0:store]=15 
            print('GOAL:  ',goal)
            label_Correction(env,agentPos,cost_grid,goal,grid_flag)
            print(grid_flag)
            print(cost_grid)
            step(env,seq[i])
            # plot_env(env)
            print('-------------------------------------------------------------')

            Q[0,i]=cost_grid[l_keypos+1,b_keypos]
            Q[1,i]=cost_grid[l_keypos-1,b_keypos]
            Q[2,i]=cost_grid[l_keypos,b_keypos+1]
            Q[3,i]=cost_grid[l_keypos,b_keypos-1]
            
            Q[4,i]=cost_grid[l+1,b]
            Q[5,i]=cost_grid[l-1,b]
            Q[6,i]=cost_grid[l,b+1]
            Q[7,i]=cost_grid[l,b-1]
            
            # Q[8,i]=cost_grid[l_doorPos,b_doorPos+1]
            Q[8,i]=cost_grid[l_doorPos,b_doorPos-1]

    return Q
#In[]     
# plot_value_function()

    
#In[]
def main():

    # env_path = './envs/example-8x8.env'
    # env_path = './envs/doorkey-5x5-normal.env'
    # env_path = './envs/doorkey-6x6-direct.env' # gif saved
    # env_path = './envs/doorkey-6x6-normal.env' # PROBLEM
    # env_path = './envs/doorkey-6x6-shortcut.env' 
    # env_path = './envs/doorkey-8x8-direct.env' # gif saved
    # env_path = './envs/doorkey-8x8-normal.env'
    env_path = './envs/known_envs/doorkey-8x8-shortcut.env' 

    env, info = load_env(env_path) # load an environment

    goal=info['goal_pos']
    agentPos=info['init_agent_pos']
    keyPos=info['key_pos']
    doorPos=info['door_pos']
    
    world_grid= (gym_minigrid.minigrid.Grid.encode(env.grid)[:,:,0].T).astype(np.float32)
    
    index= np.where(world_grid!=1 )
    world_grid[index[0][:],index[1][:]]= math.inf
    world_grid[info['key_pos'][1],info['key_pos'][0]]=-2
    world_grid[info['goal_pos'][1],info['goal_pos'][0]]=0

    world_grid[np.where(world_grid==1)]=0
    grid_flag=np.zeros(np.shape(world_grid))
    
    cost_grid=world_grid
    cost_grid[np.where(cost_grid<=0)]=0
    cost_grid_CD=cost_grid #######################################
    # print(cost_grid)

    #------------- Finding travel cost for the env When Door is closed-----------
    # Cost without door
    label_Correction(env,agentPos,cost_grid,goal,grid_flag)

    if(cost_grid[info['init_agent_pos'][1],info['init_agent_pos'][0] ] ==0):
        cost_with_door_closed=math.inf
    else:
        cost_with_door_closed=cost_grid[info['init_agent_pos'][1],info['init_agent_pos'][0] ]
    
    #-------------Fiding travel cost for the env when Door Open-------------
    world_grid= (gym_minigrid.minigrid.Grid.encode(env.grid)[:,:,0].T).astype(np.float32)
    index= np.where(world_grid!=1 )
    world_grid[index[0][:],index[1][:]]= math.inf
    world_grid[info['key_pos'][1],info['key_pos'][0]]=-2
    world_grid[info['goal_pos'][1],info['goal_pos'][0]]=0

    world_grid[np.where(world_grid==1)]=0
    grid_flag=np.zeros(np.shape(world_grid))

    cost_grid=world_grid
    cost_grid[np.where(cost_grid<=0)]=0

    world_grid[info['door_pos'][1]][info['door_pos'][0]]=0 # Remove Door from Map

    # ------------------ Finding cost to go from init_pos to Key_pos 
    label_Correction(env,agentPos,cost_grid,keyPos,grid_flag) 
    c1=cost_grid[agentPos[1],agentPos[0]]-1 # Store the cost in variable c1
    # print("C1= ",c1)
    cost_grid_OD_1=cost_grid ####################################

    # ------------
    world_grid= (gym_minigrid.minigrid.Grid.encode(env.grid)[:,:,0].T).astype(np.float32)
    index= np.where(world_grid!=1 )
    world_grid[index[0][:],index[1][:]]= math.inf
    world_grid[info['key_pos'][1],info['key_pos'][0]]=-2
    world_grid[info['goal_pos'][1],info['goal_pos'][0]]=0

    world_grid[np.where(world_grid==1)]=0
    grid_flag=np.zeros(np.shape(world_grid))

    cost_grid=world_grid
    cost_grid[np.where(cost_grid<=0)]=0

    # -----------------Finding cost to go from key to Door
    world_grid[info['door_pos'][1]][info['door_pos'][0]]=0 # Remove Door from Map
    label_Correction(env,keyPos,cost_grid,doorPos,grid_flag) 
    c2=cost_grid[keyPos[1],keyPos[0]]-1 # Store the cost in variable c2
    # print("C2= ",c2)
    cost_grid_OD_2=cost_grid #####################################

    #--------------------------
    world_grid= (gym_minigrid.minigrid.Grid.encode(env.grid)[:,:,0].T).astype(np.float32)
    index= np.where(world_grid!=1 )
    world_grid[index[0][:],index[1][:]]= math.inf
    world_grid[info['key_pos'][1],info['key_pos'][0]]=-2
    world_grid[info['goal_pos'][1],info['goal_pos'][0]]=0

    world_grid[np.where(world_grid==1)]=0
    grid_flag=np.zeros(np.shape(world_grid))

    cost_grid=world_grid
    cost_grid[np.where(cost_grid<=0)]=0

    # ----------------- Finding cost to go from key to Door
    world_grid[info['door_pos'][1]][info['door_pos'][0]]=0 # Remove Door from Map
    label_Correction(env,doorPos,cost_grid,goal,grid_flag) 
    c3=cost_grid[doorPos[1],doorPos[0]] # Store the cost in variable c3
    # print("C3= ",c3)
    cost_grid_OD_3=cost_grid #####################################
    
    

    cost_with_door_open=c1+c2+c3

    print('Cost with DOOR CLosed   ', cost_with_door_closed)
    print('Cost with DOOR Open  ', cost_with_door_open)

    ''' 
    Determine which rout will be the shortest 
    i.e. initial pose to Goal directly or 
         initial pose-> Key pose -> Door Pose -> Goal.
         Call the doorkey_problem functions accordingly .
    '''
    if(cost_with_door_closed>cost_with_door_open): 
        
        print('We need Key')
        flag=True
        # cost_grid_CD=0
    else:
        print('No key needed')
        flag=False
        # cost_grid_OD_1,cost_grid_OD_2,cost_grid_OD_3=0,0,0

    seq= doorkey_problem(flag,cost_grid_CD,cost_grid_OD_1,cost_grid_OD_2,cost_grid_OD_3,goal,agentPos,keyPos,doorPos,env,info)
    print(seq) # Get the optimal control sequence
    plot_env(env)

    env, info = load_env(env_path)
    #----------------------------------------------------------------------------
    # seq = doorkey_problem(env) # find the optimal action sequence
    # draw_gif_from_seq(seq, load_env(env_path)[0], path='./gif/example-8x8.gif') # draw a GIF & save

    # PLOT VALUE FUNCTIONS
    Q=plot_value_function(env,seq,goal,agentPos,doorPos,flag,info)
    Q[np.where(Q==math.inf)]=15
    plt.plot(Q[0,:],'--')
    plt.plot(Q[1,:],'-')
    plt.plot(Q[2,:],'--')
    plt.plot(Q[3,:],'-')
    plt.plot(Q[4,:],'--')
    plt.plot(Q[5,:],'-')
    plt.plot(Q[6,:],'--')
    plt.plot(Q[7,:],'-')
    plt.plot(Q[8,:],'--')
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Value function')
    plt.legend(["Pickup-1",
                "Pickup-2",
                "Pickup-3",
                "Pickup-4",
                "Goal_loc-1",
                "Goal_loc-2",
                "Goal_loc-3",
                "Goal_loc-4",
                "Unlock Door"],fontsize=12,loc=1)
    plt.show()
    print(Q)



#In[]
if __name__ == '__main__':
    # example_use_of_gym_env()
    main()



# %%


# %%

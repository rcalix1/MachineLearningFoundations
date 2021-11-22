## HMM with Viterbi
## tested with python 3.5

import numpy as np

#########################################################

states = ('Rainy', 'Sunny')

observations = ('walk','shop','clean','walk')

## observations = ('shop','walk','walk','clean')

## observations = ('clean', 'clean', 'clean', 'walk')

#########################################################

start_probability = {'Rainy': 0.6, 'Sunny':0.4}

transition_probability = {
          'Rainy' : {'Rainy': 0.7, 'Sunny':0.3},
          'Sunny' : {'Rainy': 0.4, 'Sunny':0.6},
    }

emission_probability = {
          'Rainy' : {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
          'Sunny' : {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
    }

#########################################################

def print_dptable(V):
    print(V)
    print("     ")
    obs = len(observations)
    sta = len(states)
    m = np.zeros((obs, sta))

    for ind_sta, state_str in enumerate(   V[0].keys()  ):
        print(ind_sta, state_str)
        for obs_t in range(len(V)):
            m[obs_t, ind_sta] = V[obs_t][state_str]

    print("  Rainy       Sunny")
    print(m)


#########################################################

def viterbi(obs,states,start_p,trans_p,emit_p):

    V = [{}]
    path = {}

    ####################################################
    ## for first observation - obs(0)
    ##
    ## V = [   {start_P[sunny]*Emit[sunny][obs(0)]    ,   start_p[rainy]*Emit[rainy][obs(0)]    },
    ##      {},
    ##      {},
    ##      {}]

    for sta_y in states:
        V[0][sta_y] = start_p[sta_y] * emit_p[sta_y][obs[0]]
        path[sta_y] = [sta_y]

    print("path")
    print(path)
    print()

    ####################################################
    ## now look at the other observations
    ## excluding obs(0) -- Notice range starts at 1 (not 0)
    ## obs_t here in integer index for the observations    

    for obs_t in range(1, len(obs)):
        V.append({})
        newpath = {}
        
        ## at current observation, calculate prob per state
        for st_y in states:
            list_prob_state = []
            for st_y0 in states:
                ##         prev_obs     state-to-state          state-to-obs
                prob = V[obs_t-1][st_y0] * trans_p[st_y0][st_y] * emit_p[st_y][obs[obs_t]]
                state_temp = st_y0

                list_prob_state.append(    (prob, state_temp)     )
            
            (prob, state) = max(     list_prob_state      )     

            V[obs_t][st_y] = prob
            newpath[st_y] = path[state] + [st_y]
        path = newpath
        print(path)
    ###########################################
    ## print V dictionary

    print_dptable(V)
    print()

    ###########################################

    list_prob_state = []
    ## gets probs of last observation for each state
    ## and picks max
    for st_y in states:
        
        prob_temp = V[   len(obs)-1    ][st_y]
        state_temp = st_y
        list_prob_state.append(    (prob_temp, state_temp)     )

    (prob, state) = max(          list_prob_state             )

    print(path)
    return (prob, path[state])


#########################################################

def example1():
    return viterbi(observations,states,start_probability,transition_probability,emission_probability)


#########################################################

##Main()

print(observations)

print(   example1()  )

print('<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>')



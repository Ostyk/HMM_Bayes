import numpy as np

class MyHMM(object):
    '''Hidden Markov Model implementation'''

    def __init__(self,model_name):
        '''
        T -- transition model probabilities
        O -- observation (sensor) model probabilities
        pi -- initial probabilities
        '''
        to_mat = lambda x: np.vstack((np.array(x),np.array(x)[::-1]))
        self.T = to_mat(model["T"])
        self.O = np.diag(model["O"])
        self.pi = np.array(model["pi"])
        print("transition model:\n{}\nObservation Model:\n{}\nInitial probabilities:\n{}".format(self.T,self.O,self.pi))
        return None

    def sample(self, N_seq, len_):
        '''
        args:
        N_seq - number of sequences to be stored
        len_ - length of sequences
        returns:
        states - matrix of size (N_seq,len_)
        observations - matrix of size (N_seq,len_)
        '''
        states = np.zeros((N_seq, len_), dtype=np.int)
        observations = states.copy()
        rain = 0
        for sequence in range(N_seq):
            R_t = np.random.binomial(n=1, p=self.pi[0]) #R_(t-1) -- first day
            if R_t == 1: rain+=1
            for i in range(len_):
                if R_t==0: #no rain
                    U_t = np.random.binomial(n=1, p=self.O[1,1])
                    R_t = np.random.binomial(n=1, p=self.T[0,0])
                elif R_t==1: # rain
                    U_t = np.random.binomial(n=1, p=self.O[0,0])
                    R_t = np.random.binomial(n=1, p=self.T[1,1])
                states[sequence,i]=R_t
                observations[sequence,i]=U_t

        print("\nRain on the first day: {}/{} times".format(rain, N_seq))
        return states, observations

    def forward(self, n, k, obs):
        '''forward part
        args:
        n - number of states
        k - length of observation vector
        '''
        
        fw = np.zeros((n,k+1)) #init row vector at time 0
        fw[:, 0] = self.pi #prior information
        for obs_ind in range(k): #propagation
            f_row_vec = np.matrix(fw[:,obs_ind])
            fw[:, obs_ind+1] = f_row_vec * \
                               np.matrix(self.T) * \
                               np.matrix(np.diag(self.O[:,obs[obs_ind]])) 
                    #current estimate
            fw[:, obs_ind+1] /= np.sum(fw[:,obs_ind+1]) # normalize and store
        return np.array(fw)

    def backward(self, n, k, obs):
        '''backward part
        This is the same as the forward algorithm 
        except that it start at the end and works 
        backward toward the beginning. 
        '''
        
        bw = np.zeros((n,k+1))
        bw[:,-1] = 1.0 #vector of ones smoothing part
        for obs_ind in range(k, 0, -1): #backwards propagation
            b_col_vec = np.mat(bw[:,obs_ind]).T
            bw[:, obs_ind-1] = (np.matrix(self.T) *
                                np.matrix(np.diag(self.O[:,obs[obs_ind-1]])) *
                                b_col_vec).T
            bw[:,obs_ind-1] /= np.sum(bw[:,obs_ind-1]) # normalize and store
        return np.array(bw)

    def forward_backward(self,obs):
        '''forward-backward algorithm'''
        n, k = self.O.shape[0], obs.size
        fw = self.forward(n, k, obs)
        bw = self.backward(n, k, obs)
        prob_mat = fw * bw # element wise multiplication
        prob_mat /= np.sum(prob_mat, 0) #normalizing because Bayes..
        return prob_mat, fw, bw


def state_check(ind, current_observation, probabilities):
    correct = np.sum(current_observation == probabilities[1,:][1:])
    l = len(current_observation)
    space = " "*(3-len(str(ind)))
    print("\ts{}:{}{}/{}".format(ind,space,correct,l))

if __name__ == '__main__':
    model = {"T":[0.7,0.3],
             "O":[0.9,0.2],
             "pi":[0.5,0.5]}
    N_sequences = 15
    length = 20

    M = MyHMM(model)
    states, observations = M.sample(N_sequences, length)
    #Hidden no rain = 0 or rain = 1
    #observation no umbrella = 0 or umbrella = 1

    print("\nNumber of correct estimates for each sequence S:")
    for i in range(N_sequences):
        current_state = states[i]
        #actual probability of each state at each time step of our process, given the observations
        probabilities, fw, bw = M.forward_backward(observations[i])
        state_check(i, current_state, probabilities)

##############################
# Result of running the file #
##############################

# transition model:
# [[0.7 0.3]
#  [0.3 0.7]]
# Observation Model:
# [[0.9 0. ]
#  [0.  0.2]]
# Initial probabilities:
# [0.5 0.5]
#
# Rain on the first day: 5/15 times
#
# Number of correct estimates for each sequence S:
# 	s0:  12/20
# 	s1:  8/20
# 	s2:  15/20
# 	s3:  14/20
# 	s4:  10/20
# 	s5:  12/20
# 	s6:  10/20
# 	s7:  12/20
# 	s8:  12/20
# 	s9:  14/20
# 	s10: 6/20
# 	s11: 12/20
# 	s12: 12/20
# 	s13: 14/20
# 	s14: 15/20

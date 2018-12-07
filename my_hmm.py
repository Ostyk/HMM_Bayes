import numpy as np

class MyHMM(object):
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
            R_t = np.random.binomial(n=1, p=self.pi[0])
            if R_t == 1: rain+=1
            for i in range(len_):
                if R_t==0:
                    U_t = np.random.binomial(n=1, p=self.O[1,1])
                    R_t = np.random.binomial(n=1, p=self.T[0,0])
                elif R_t==1:
                    U_t = np.random.binomial(n=1, p=self.O[0,0])
                    R_t = np.random.binomial(n=1, p=self.T[1,1])
                states[sequence,i]=R_t
                observations[sequence,i]=U_t

        print("\nRain on the first day: {}/{} times".format(rain,N_seq))
        return states, observations

    def forward(self, n, k, obs):
        '''forawrd part'''
        fw = np.zeros((n,k+1))
        fw[:, 0] = self.pi #prior information
        for obs_ind in range(k):
            f_row_vec = np.matrix(fw[:,obs_ind])
            fw[:, obs_ind+1] = f_row_vec*np.matrix(self.T)*np.matrix(np.diag(self.O[:,obs[obs_ind]]))
            fw[:, obs_ind+1] /= np.sum(fw[:,obs_ind+1]) # normalize and store
        return np.array(fw)

    def backward(self, n, k, obs):
        '''forawrd part'''
        bw = np.zeros((n,k+1))
        bw[:,-1] = 1.0 #vector of ones smoothing part
        for obs_ind in range(k, 0, -1):
            b_col_vec = np.mat(bw[:,obs_ind]).T
            bw[:, obs_ind-1] = (np.matrix(self.T)*np.matrix(np.diag(self.O[:,obs[obs_ind-1]]))*b_col_vec).T
            bw[:,obs_ind-1] /= np.sum(bw[:,obs_ind-1]) # normalize and store
        return np.array(bw)

    def forward_backward(self,obs):
        '''forward-backward algorithm'''
        n, k = self.O.shape[0], obs.size
        fw = self.forward(n,k, obs)
        bw = self.backward(n,k,obs)
        prob_mat = fw * bw
        prob_mat /= np.sum(prob_mat, 0) #normalizing because Bayes..
        return prob_mat, fw, bw


def observation_check(ind, current_observation, probabilities):
    correct = np.count_nonzero(current_observation == probabilities[1,:][1:])
    l = len(current_observation)
    print("\tS_{}: {} / {}".format(ind,correct,l))

if __name__ == '__main__':
    model = {"T":[0.7,0.3],
             "O":[0.9,0.2],
             "pi":[0.5,0.5]}
    N_sequences = 15
    length = 20

    M = MyHMM(model)
    states, observations = M.sample(N_sequences, length)

    print("\nNumber of correct estimates for each sequence S:")
    for i in range(N_sequences):
        current_observation = observations[:,i]
        probabilities, fw, bw = M.forward_backward(current_observation)
        observation_check(i, current_observation, probabilities)

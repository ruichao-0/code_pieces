r"""
CTC (Connectionist Temporal Classification).

Author: Ruichao Jiang
-------------------------------------------

The following is my understanding of the paper CTC

This paper wants to deal with a problem: How to map sequences of input signals
to sequences of letters. The length of a sequence is not specified but assumed
to be less than the length of signals. The space $Z$ is a space of strings, a
path space (like Wiener path space or Feynman path space).

Inference: The paper transform the network output into a measure on the path
           space, conditioned on the input. Hence it is a random measure, or
           a transition kernel if it admits a density. This random measure can
           be reduced to a random varaible $h:X\to Z$, by selecting the output
           correspondig to the maximum conditional probability.
           
Simplifications/Assumptions: 
           1. A functional $LER(h,S')$ is defined. It simplifies the problem
           of estimating a random measure to a variational problem: finding
           a measure is reduced to minimising a functional.

           2. Use neural network to limit the possible form of $h$. It is like
           using Fourier expansion to approximate a function. Choosing a neural
           network is like deciding how many plane waves we'd like to use.

           3. Once the architecture has been fixed, it remains to train the
           weights, like calculating Fourier coefficients. This is done by
           minimising the negative log likelihood.
------------------------------------------------------------------------------          

The following is simply wrong.

Eqn (2) transforms the output of the network as conditional measure. It is a
product measure, which means (conditional) independence The authors recognize
it as an assumption. But they further say that this assumption is ensured by 
no feedback. This is DEFINITELY FALSE. It reamins an assumption.
"""

import numpy as np

blank = '0'
alphabates = {
    'a' : 0, 'b' : 1, 'c' : 2, 'd' : 3, 'e' : 4,
    'f' : 5, 'g' : 6, 'h' : 7, 'i' : 8, 'j' : 9,
    'k' : 10, 'l' : 11, 'm' : 12, 'n' : 13, 'o' : 14,
    'p' : 15, 'q' : 16, 'r' : 17, 's' : 18, 't' : 19,
    'u' : 20, 'v' : 21, 'w' : 22, 'x' : 23, 'y' : 24,
    'z' : 25, blank : 26
    }

anti_alphabates = {
    0 : 'a', 1 : 'b', 2 : 'c', 3 : 'd', 4 : 'e',
    5 :'f', 6 : 'g', 7 : 'h', 8 : 'i', 9 : 'j',
    10 : 'k', 11 : 'l', 12 : 'm', 13 : 'n', 14 : 'o',
    15 : 'p', 16 : 'q', 17 : 'r', 18 : 's', 19 : 't',
    20 : 'u', 21 : 'v', 22 : 'w', 23 : 'x', 24 : 'y',
    25 : 'z', 26 : blank
    }

class CTC:
    r"""
    A class to represent a CTC layer.
    
    ...
    
    Attributes
    ----------
    l: string, predicted labbeling by the network  l = B(\pi) as in paper
    network_output: T x 27 matrix of probabilities computed by the network,
                    where T is the total time and 27 is the cardinality of the
                    alphabates.
    path_probability: double, Pr(l|x) = \sum_{\pi\in\B^{-1}(l)} Pr(\pi|x)
    gradient: np.array, gradient of Pr(l|x)
    path : string, Best path decoding path = B(argmax Pr(\pi|x))
    """
    
    def __init__(self, l, network_output):
        self.l = l
        self.network_output = network_output
        self.path_probability = self.forward_backward()[0]
        self.gradient = self.forward_backward()[1]
        self.path = self.best_path()

    def forward_backward(self):
        r"""Calculate path_probability and gradient.
        
        Helper functions:
        \alpha_t(s) = \sum_{B(\pi_{1:t}=l_{1:s})}\product_{t'}y^t'_\pi_t'
        \beta_t(s) =                                                    

        Use dynamical programming alpha_cache and beta_cache
        """
        T = self.network_output.shape[0]
        l_prime = '^%s^' % '^'.join(self.l)
        l_prime_len = len(l_prime)
        

        alpha_cache = np.zeros((T, l_prime_len))
        
        def alpha(t, s):
            r"""
            Calculate helper function \alpha_t(s).
            
            Parameters
            ----------
            t : int
                current time, from 0 to T - 1.
            s : int
                current location in l', from 0 to l_prime_len - 1.

            Returns
            -------
            double
                \alpha_t(s), to be stored in the dynamical programming cache.

            Base
            ----
            \alpha_0(0) = y^0_blank
            \alpha_0(1) = y^0_l_1
            \alpha_0(s) = 0 otherwise

            Recursion
            ---------
            If l'_s = blank or l'_{s-2} = l'_s,
            \alpha_t(s) = (\alpha_{t -1}(s) + \alpha_{t - 1}(s - 1)) * y^t_l'_s

            Otherwise,
            \alpha_t(s) = (\alpha_{t - 1}(s) + \alpha_{t - 1}(s - 1) +
                           \alpha_{t - 1}(s - 2)) * y^t_l'_s
            """
            if t == 0:
                if s == 0:
                    return self.network_output[0][alphabates[blank]]
                elif s == 1:
                    return self.network_output[0][alphabates[l_prime[1]]]
                else:
                    return 0

            if alphabates[s] == blank or (s >= 2 and alphabates[s] == alphabates[s - 2]):
                return (alpha_cache[t - 1][s] + alpha_cache[t - 1][s - 1]) * self.network_output[t][alphabates[l_prime[s]]]
            else:
                return (alpha_cache[t - 1][s] + alpha_cache[t - 1][s - 1] + alpha_cache[t - 1][s - 2]) * self.network_output[t][alphabates[l_prime[s]]]
            
        for t in range(T):
            for s in range (l_prime_len):
                alpha_cache[t][s] = alpha(t, s)
                
        beta_cache = np.zeros((T, l_prime_len))
        
        def beta(t, s):
            r"""
            Calculate helper function \beta_t(s).

            Parameters
            ----------
            t : int
                Backward current time, from T - 1 to 0.
            s : int
                Backward current location in l', from l_prime_len - 1 to 0

            Returns
            -------
            double
                \alpha_t(s), to be stored in the dynamical programming cache.
                
            Base
            ----
            \beta_{T - 1}(l_prime_len - 1) = y^{T - 1}_blank
            \beta_{T - 1}(1_prime_len - 2) = y^0_l_1
            \beta_{T - 1}(s) = 0 otherwise

            Recursion
            ---------
            If l'_s = blank or l'_{s + 2} = l'_s,
            \beta_t(s) = (\beta_{t + 1}(s) + \beta_{t + 1}(s + 1)) * y^t_l'_s

            Otherwise,
            \beta_t(s) = (\beta_{t + 1}(s) + \beta_{t + 1}(s + 1) +
                           \beta_{t + 1}(s + 2)) * y^t_l'_s

            """
            if t == T - 1:
                if s == l_prime_len - 1:
                    return  self.network_output[-1][alphabates[blank]]
                elif s == l_prime_len - 2:
                    return self.network_output[-1][alphabates[l_prime[-2]]]
                else:
                    return 0

            if alphabates[s] == blank or (s >= 2 and alphabates[s] == alphabates[s - 2]):
                return (beta_cache[t + 1][s] + beta_cache[t + 1][s + 1]) * self.network_output[t][alphabates[l_prime[s]]]
            else:
                return (beta_cache[t + 1][s] + beta_cache[t + 1][s + 1] + beta_cache[t + 1][s + 2]) * self.network_output[t][alphabates[l_prime[s]]]
            
        for t in range(T - 1, -1, -1):
            for s in range(l_prime_len - 1, -1, -1):
                beta_cache[t, s] = beta(t, s)
        
        def gradient():
            gradient = np.zeros_like(self.network_output)
            for t in range(T):
                for k in range(27):
                    for s in range(len(self.l)):
                        if self.l[s] == anti_alphabates[k]:
                            gradient[t][k] += alpha_cache[t][s] * beta_cache[t][s]
                    gradient[t][k] /= (self.network_output[t][k] ** 2)
                    gradient[t][k] *= -1
            return gradient
        
        return alpha_cache[-1][-1] + alpha_cache[-1][-2], gradient


    def best_path(self):
        """Find the most probable sequence (only approxinately)."""
        most_probable_ind = np.argmax(self.network_output, axis = 1)
        path = anti_alphabates[most_probable_ind[0]]
        for t in range(1, self.network_output.shape[0]):
            if path[-1] != anti_alphabates[most_probable_ind[t]]:
                path += anti_alphabates[most_probable_ind[t]]
        return path.replace('0','')

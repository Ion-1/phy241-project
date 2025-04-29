import numpy as np

def nnl(args):
    mean_k,sigma_k,sigma_pi=args
    mean_pi=1
    amount_pi=0.84
    amount_k=1-amount_pi
    data=np.loadtxt(".\data\dec_lengths.txt")
    probs

import numpy as np
import scipy
from functools import partial

def nll(data, mean_k):
    #print("testing at", mean_k)
    probs=np.clip(distribution(mean_k,data),1e-10,1)
    logs=np.log(probs)
    return -np.sum(logs)

def distribution(mean_k,x):
    mean_pi = 4188
    amount_pi = 0.84
    amount_k = 1 - amount_pi
    return amount_k*scipy.stats.expon.pdf(x, loc=0, scale=mean_k) + amount_pi*scipy.stats.expon.pdf(x, loc=0, scale=mean_pi)

def task():
    data = np.loadtxt(".\data\dec_lengths.txt")
    start=500
    bounds = (0,5000)
    best = scipy.optimize.minimize(partial(nll, data), start)
    return best

print(task())


from os import write
import numpy as np 
import coms
from forward import s3
import os

def wcoordinates(theta, e):
    filename = 'outarray.npy'
    path = os.path.dirname(os.path.abspath(__file__)) + "/" + filename
    print(path)
    try:
        f = open(path, 'r')
        f.close()
    except:
        for x in range(theta.shape[0]):
            print("%",100*(x/theta.shape[0]))
            for y in range(theta.shape[0]):
                for z in range(theta.shape[0]):
                    s = s3(theta[x], theta[y], theta[z])
                    # print(int(s[0]),int(s[1]))
                    if( (int(s[0]) >= 0) and (int(s[1]) >= 0) ):
                        e[int(s[0])][int(s[1])] = [theta[x], theta[y], theta[z]]
            
        np.save(path, arr=e)

def rcoordinates():
    filename = 'outarray.npy'
    path = os.path.dirname(os.path.abspath(__file__)) + "/" + filename
    print(path)

    return np.load(path)



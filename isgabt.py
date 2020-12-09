import numpy as np
import math

# Constants
pi = math.pi

# Population Size
N = 1000

# Target Matrix
X = ['x', 'y', 'z']

# Joint Variable Encoding - Dtermines resolution power and size of search space.
theta1 = pi/4
theta2 = (3*pi)/4
theta3 = pi

Bin = []

# Initial Joint Value
Vs = 0

# Initial Terminal Joint Value
Vt = 0

# Number of point in trajectory
Nt = 0

# Reference Points
sp = (Vs - Vt)/(Nt -1)
Rj = Vs - (Nt -1) * sp

# Initial Desired Posture
...


def isgabt(PopulationSize, TargetMatrix):
    return #Optimal Solution
    
def searchrange(Population, TargetMatrix, PreviousSolutions, SearchRadius):
    return S #Search Interval
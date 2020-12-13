#https://circuitdigest.com/microcontroller-projects/arduino-python-tutorial
#https://pynput.readthedocs.io/en/latest/mouse.html#reference

from deap.tools.support import HallOfFame
from numpy.core.defchararray import add, array, multiply
from numpy.core.function_base import linspace
from numpy.core.numeric import full
import numpy as np 
import coms
from forward import s3,s2,s1
import math
from math import exp, radians as rad, sqrt
from math import degrees as deg
from deap import creator
from deap import gp 
from deap import base
from deap import tools
from deap import algorithms
import operator
import random
import operator

# Begin Arduino programming via sketch.
# print("Load Arduino Sketch \"setupsketch.ino\" ? (y/n)")
# keypress = input()
# if keypress == ord("y"):
#     print("Loading sketch...")
#     coms.upload()

#Initialize serial
try:
    com = coms.dueserial()
except:
    print("\n Unable to connect to serial device.")

# Serial write packet.
data_out = [10, 90, 45, 180 ,180, 90, 10]

# Robot Dimensions.
# 1st arm from shoulder, forecep, wrist with gripper length.
a = np.array([12.5, 12.5, 18.5])

# Physical Constraints
# 0 is shoulder
# 1 is elbow
# 2 is Wrist V
# 4 is Wrist R
# 5 is Grabber

base_min = 0.0
base_max = 180.0
shoulder_min = 15.0
shoulder_max = 165.0
elbow_min = 0.0
elbow_max = 180.0
wristv_min = 0.0
wristv_max = 180.0
wristr_min = 0.0
wristr_max = 180.0
gripper_min = 10.0
gripper_max = 73.0

constraint = np.array([  [shoulder_min, shoulder_max],             
                            [elbow_min, elbow_max],                 
                            [wristv_min, wristv_max],           
                            [wristr_min,wristr_max],        
                            [gripper_min, gripper_max]  ])

# Initialize Learning Base
# theta = np.array([
#     [0.0, 0.0, 0.0], 
#     [45.0, 0.0, 0.0], 
#     [90.0, 0.0, 0.0], 
#     [135.0, 0.0, 0.0], 
#     [180.0, 0.0, 0.0],
#     [0.0, 180.0, 0.0],
#     [0.0, ]
#     [180.0, 180.0, 180.0]])

theta = np.zeros((np.linspace(0,180,10).shape[0], 3), dtype=float)
oneeighty = np.linspace(0, 180, 10)

for x in range(theta.shape[0]):
    for y in range(theta.shape[1]):
        theta[x][theta.shape[1]-1] = oneeighty[x]
        theta[x][theta.shape[1]-2] = oneeighty[theta.shape[0]-x-1]
        theta[x][theta.shape[1]-3] = oneeighty[x]

print(theta)

# theta = np.empty((100,3), dtype=float)

# for x in range(theta.shape[0]):
#     theta[x] = [random.uniform(0,180),random.uniform(0,180),random.uniform(0,180)]

# Extract single dimension values for x y values from transform
s3 = np.array([s3(theta[i][0], theta[i][1],theta[i][2]) for i in range(theta.shape[0])])
s2 = np.array([s2(theta[i][0], theta[i][1],theta[i][2]) for i in range(theta.shape[0])])
s1 = np.array([s1(theta[i][0], theta[i][1],theta[i][2]) for i in range(theta.shape[0])])
print(s1)
s = np.array([s1, s2, s3])
print((s[0])[1][1])
# Transform degrees to radians    
for i in range(theta.shape[0]):
    for j in range(theta.shape[1]):
        theta[i][j] = rad(theta[i][j])

# vx = np.zeros(results.shape[0])
# vy = np.zeros(results.shape[0])

# for i in range(results.shape[0]):
#     vx[i] = results[i][0]
#     vy[i] = results[i][1]

exception = False

# Initialize Terminal and Function Set
def protectedDiv(left, right):
    try:
        exception = False
        return left / right
    except ZeroDivisionError:
        exception = True
        return 100000

def protectedAsin(left):
    try:
        exception = False
        return math.asin(left)
    except:
        exception = True
        return 100000
def protectedAcos(left):
    try:
        exception = False
        return math.acos(left)
    except:
        exception = True
        return 100000
def protectedAtan(left):
    try:
        exception = False
        return math.atan(left)
    except:
        exception = True
        return 100000
def protectedSqrt(left):
    try:
        exception = False
        return math.sqrt(left)
    except:
        exception = True
        return 100000
def protectedCos(left):
    try:
        exception = False
        return math.cos(left)
    except:
        exception = True
        return 100000
def protectedSin(left):
    try:
        exception = False
        return math.sin(left)
    except:
        exception = True
        return 100000
def protectedTan(left):
    try:
        exception = False
        return math.tan(left)
    except:
        exception = True
        return 100000
def     mul(left, right):
    return operator.mul(left, right)
def sub(left, right):
    return operator.sub(left,right)
def add(left,right):
    return operator.add(left,right)
def linklength(angle):
    if angle == 0:
        return 12.5
    if angle == 1:
        return 12.5
    if angle == 2:
        return 18.5
    else:
        return 1

fset = gp.PrimitiveSet("MAIN", 4)
fset.addPrimitive(add, 2)
fset.addPrimitive(sub, 2)
fset.addPrimitive(mul, 2)
fset.addPrimitive(protectedDiv, 2)
# fset.addPrimitive(protectedAcos, 1)
# fset.addPrimitive(protectedCos, 1)
# fset.addPrimitive(protectedSin, 1)
fset.addPrimitive(protectedAsin, 1)
fset.addPrimitive(protectedTan, 1)
fset.addPrimitive(protectedAtan, 1)
fset.addPrimitive(protectedSqrt, 1)
fset.addPrimitive(linklength, 1)
fset.addEphemeralConstant("rand101", lambda: random.uniform(-1.0,1.0))
fset.renameArguments(ARG0="x")
fset.renameArguments(ARG1="y")
fset.renameArguments(ARG2="angle")
fset.renameArguments(ARG3="prevangle")


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=fset, min_=1, max_=4)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=fset)

x = -43.5
y = 0
prev =  np.array([([0],[0],[0]) for i in range(theta.shape[0])])
prevangle = 0

for angle in range(0,3):

    def evalSymbReg(individual, points):
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=individual)
        # Evaluate the mean squared error between the expression
        # and the real function : x**4 + x**3 + x**2 + x
        sqerrors = 0.0
        # sqerrors = np.zeros(points.shape[0])
        for i in range(points.shape[0]):
            sqerrors = sqerrors + ((func(points[i][0]-prev[i][0], points[i][1]-prev[i][1], linklength(angle), prevangle) - (theta[i][angle]))**2.0)
            # print("pointsx:", points[i][0]-prev[i][0], "y:", points[i][1]-prev[i][1], "Link:", linklength(angle), angle, "angle", deg(theta[i][angle]))
            if(0.00 <= func(points[i][0], points[i][1], angle, prevangle) <= 180.0):
                sqerrors = sqerrors + 10000
            elif(func(points[i][0], points[i][1], angle, prevangle) > 180):
                sqerrors = sqerrors + 10000
        #    print(sqerrors)
        return (sqerrors / len(points)),
    
    def feasible(individual):
        if(exception == True):
            return False
        else:
            return True
        
    toolbox.register("evaluate", evalSymbReg, points=s[angle])
    toolbox.register("select", tools.selTournament, tournsize=20)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=fset)
    toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 7.0))
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    # Initialize Population
    pop = toolbox.population(n=2000)
    hof = tools.HallOfFame(1)
    pop, log = algorithms.eaSimple(pop, toolbox, .7, 0.3, 20,
                                    halloffame=hof, verbose=False)

    print(hof[0], hof[0].fitness, hof[0].fitness.valid)

    angleinrad = eval(str(hof[0]))
    print("angle%d=" % angle, deg(angleinrad))

    prev = s[angle]
    prevangle = angleinrad


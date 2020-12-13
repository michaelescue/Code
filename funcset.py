
import operator
import math

# Initialize Terminal and Function Set
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def protectedAsin(left):
    try:
        return math.asin(left)
    except:
        return 1
def protectedAcos(left):
    try:
        return math.acos(left)
    except:
        return 1
def protectedAtan(left):
    try:
        return math.atan(left)
    except:
        return 1
def protectedSqrt(left):
    try:
        return math.sqrt(left)
    except:
        return 1
def protectedCos(left):
    try:
        return math.cos(left)
    except:
        return 1
def protectedSin(left):
    try:
        return math.sin(left)
    except:
        return 1
def protectedTan(left):
    try:
        return math.tan(left)
    except:
        return 1
def     mul(left, right):
    return operator.mul(left, right)
def sub(left, right):
    return operator.sub(left,right)
def add(left,right):
    return operator.add(left,right)
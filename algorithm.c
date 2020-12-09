/*
p0 = shoulder base.
p1 = elbow
p2 = wrist
pe = End effector (Grabber)
tagetx, targety = pe target (can develop from spherical coordinates)

Goal is axis aligment along known end effector coordinate from all three origins.
Once an origin is at its rotation bound, it is skipped.

Endeff coordinates are known (x,y).

IF xee, yee == target x,y skip code section.

Else Begin iteration (loop) between all origins, starting with p0.

If origin rotation is at max, break.
    p0 example:

    New p angle = Current p angle + ThetaDelta
        p theta = (p theta) + (arctan(ytarg/xtarg) - arctan(yee,xee) + sum(pn-1 to p0))
   
    If calculated theta exceeds bound
        Set p angle to exceeded bound value
    
    Update next origin coordinates

    Update Endeff coordinates (use r eff, theta eff)
    
    Move to next origin (break)

*/
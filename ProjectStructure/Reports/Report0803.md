## Report 08/03/2023

The bundle adjustement is the optimization of the location of the 3d points

th camera is calibrated but the extrinsic parameter of the camera need to be optimized


in order to avoid the accumulation of the error, we need to bundle every 5-10 frames (window bundle adjustement), but also bundle adjustement after every frame, in the end (to begin with)

1. get a slam implementation with BA on the entire sequence


PIcking out a sequence of 20 frames and work on those doing bundle adjustement

input: initial guesses
output: optimized refined poses

function

cluster features

LOOP CLOSURE

U WOULD store an history of the entire sequence and when we see a new frame we can scroll the history, we canset a threshold and make a pose estimation between the historic frame.


three groups:

BA
VO
Feature detection

     
matrix that orders everything by point and by frames
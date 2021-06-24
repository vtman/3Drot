# 3Drot
Creating of a list of optimal 3D rotations

We generate a list of optimal 3D rotations. Each new rotation from the list is found in a way to minimise the 
gap in angular coverage for a global angular search or a local angular refinement problem.
Rotations can be combined in groups for some problems. 
The algorithm also works for those combined rotations. 

<b>Global problem.</b> We aim to find all possible rotations (there is no restrictions). The main application is to pre-rotate 3D datasets (as cuboids). 
In a general  case an object can be pre-rotated 4 times.

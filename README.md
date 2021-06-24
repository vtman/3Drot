# 3Drot
Creating of a list of optimal 3D rotations

We generate a list of optimal 3D rotations. Each new rotation from the list is found in a way to minimise the 
gap in angular coverage for a global angular search or a local angular refinement problem.
Rotations can be combined in groups for some problems. 
The algorithm also works for those combined rotations. 

<b>Global problem.</b> We aim to find all possible rotations (there is no restrictions). The main application is to pre-rotate 3D datasets (as cuboids). 
In a general  case an object can be pre-rotated 4 times.

<img src="images/group4.png" width="700">

For each new rotation matrix we get group of 4 matrices using the matrices shown above.

If an object has two same dimensions, e.g. width and height are the same, then get rotations combined in groups of 8.

<img src="images/group8.png" width="700">

When all dimensions are the same, then we get 24 rotations in a group.

<img src="images/group24.png" width="700">

The code (quatGlobal.cpp) allows a user to generate the list of rotations. The code uses OpenMP and requries the use of Intel Compiler (can be obtained from <a href="https://software.intel.com/content/www/us/en/develop/tools/oneapi/all-toolkits.html">oneAPI</a>).

Parameters to modify.


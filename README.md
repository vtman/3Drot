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

<hr>

The code (quatGlobal.cpp) allows a user to generate the list of rotations. The code uses OpenMP and requries the use of Intel Compiler (can be obtained from <a href="https://software.intel.com/content/www/us/en/develop/tools/oneapi/all-toolkits.html">oneAPI</a>).

Parameters to modify.
<ul>
  <li>Output text file (line 491).</li>
  <li>Output file for values of angular distances (3D cube, line 449). The function <tt>writeOutput</tt> is commented out (to uncomment see line 868).</li>
  <li>Number of items in the list. Currently it is set to 100 (line 806).</li>
  <li>Number of rotations in a group (<tt>ng</tt>, line 15). When this parameter is changed, you also need to specify how those rotations are generated. Examples of those rotations are provided in <tt>createRotMat</tt> function.</li>
  <li>Size of grids used to generate rotations (parameters <tt>block_sizeh</tt> in line 20 and <tt>nblocks</tt> line 22). The first parameter controls the size of blocks used by each thread, the second one is to find the total number of blocks processed by all threads.</li>
</ul>

Output file. Each line defines a rotation (only one rotation within a group, all other rotations can be found using the known rotation matrices)
<ul>
  <li>Index of a rotation.</li>
  <li>Four components of the corresponding quaternion.</li>
  <li>An estimate of the maximum distance between an arbitary rotation and the nearest rotation from the list (all rotations before and including the new one).</li>
  <li>The corresponding angular value for this distance (in degrees).</li>
</ul>

<hr>
  
<b>Local problem.</b> Suppose there is a function defined for any possible rotation/orientation. A user may usually need to find those rotations when the function attains its minimum/maximum value. A list of rotations obtained for a global problem can be used to roughly estimate those solutions. Once the best rotations are roughly estimated we need to refine them. So, we just need to consider rotations around given ones (within some angular distance). For the local problem we specify this angular distance and generate a list of optimal rotations such that for any arbitarary rotation within the given angular distance its distance to the nearest rotation within the list is ever decreasing.

Code (quatLocal.cpp) is used for the local problem.

Parameters to modify.
<ul>
  <li>Output text file (line 387).</li>
  <li>Number of items in the list. Currently it is set to 100000 (line 785).</li>
  <li>Initial angular distance (in degrees, line 377).</li>
  <li>Size of grids used to generate rotations (parameters <tt>block_sizeh</tt> in line 18 and <tt>nblocks</tt> line 20).
</ul>


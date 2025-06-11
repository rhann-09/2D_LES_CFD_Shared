Sharing my messing around with the code from: https://github.com/pranshupant/2D_LES_CFD

Their original implementation is detailed in guo_joseph_pant_report.pdf

Original Authors:

⋅⋅* @pranshupant * Pranshu Pant
⋅⋅* @joejoseph007 * Joe Joseph
⋅⋅* @jguo4

Changes from Original Code:

Cleaned up code a bit

Conglomerated the original code into single script to be ran in Spyder IDE rather than command line

Edited dependancy on custom yaml module to inbuilt python yaml module for input file

Included automated example of writing/reading yaml file for reading inputs

Sped up I/O

Setup Sphinx Docstrings


Next Steps:


Include setting of Smagorinsky Constant in input yaml file

Further Vectorise Main Script Functions

Parallelise Run Script using the module Joblib Parallel

Port over to C++ and/or Cuda

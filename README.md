# JOC-OptimizeTrainedNNEnsemble
This repository contains the test instances and the code in the article "Optimizing over an ensemble of trained neural networks", authored by Keliang Wang, Leonardo Lozano, Carlos Cardonha, and David Bergman.
# Dataset
The **Dataset** folder contains training datasets for four benchmark functions(stored in numpy array) Peaks, Beale, Perm and Spring, and two real-world datasets Winequality_red(.csv) and Concrete_Data(.xls).
# Instances
Each instance consists of parameters of trained NN for the corresponding problem.
## Instances for comparing solution quality
- **SolutionQualityComparison** folder contains instances of four benchmark functions and are used to report results in Section 5.2.
## Instances for evaluating optimization algorithm
- **OptAlgoEvaluation** folder contains instances of four benchmark functions and two real-world problem, and are used to report results in Section 5.3 and Section 5.4. 
# Code
- **BranchCutAlg** contains source code of baseline branch and cut algorithm.
- **twoPhaseAlg** contains source code of two-phase algorithm developed in the paper.
- **twoPhaseSensitivity** contains source code for conducting sensitivity analysis.
# Results
- **OptimizationAlgorithmResults** contains .xlsx file logging the computational results reported in section 5.3 and 5.4.
- **SolutionQualityResults** contains .xlsx files logging the results of solution quality comparison reported in section 5.2.

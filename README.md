# physgp
A package for implementing physics-informed gaussian process (GP) regression for general well-posed linear PDE problems. 

Physics-informed GP regression is a powerful tool for solving linear PDEs, obtaining not only the spectral approxmation for the solution (mean function) but also a measure of uncertainty across the solution. Previous work has mainly implemented this method within MATLAB and is often difficult to implement and covariance functions must be calculated and specified for each linear PDE problem and chosen prior kernel for the solution, manually. Hence we introduce physgp, a module where we can simply specify the starting building blocks (the linear differential operator, boundary conditions, etc.) and produce a GP regression fit fast and simply. 



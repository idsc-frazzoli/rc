

# Dependencies

## main routine

None

## karma_nash_equilibrium routine

Multiprod v1.37.0.0 - https://www.mathworks.com/matlabcentral/fileexchange/8773-multiple-matrix-multiplications-with-array-expansion-enabled

Vector algebra v1.3.0.0 - https://www.mathworks.com/matlabcentral/fileexchange/8782-vector-algebra-for-arrays-of-any-size-with-array-expansion-enabled

## karma_social_welfare routine

Dependencies for karma_nash_equilibrium routine plus:

YALMIP v16-January-2020 - https://yalmip.github.io/

OPTI Toolbox v2.28 - https://inverseproblem.co.nz/OPTI/


# Instructions

## Running a simulation

1. Select parameters in load_parameters.m.
1.1. Multiple alpha values can be specified as a vector
1.2. Only one k_ave value can be specified
2. Run main.m

Note: main.m uses Nash equilibrium and/or social welfare computation results stored in karma_nash_equilibrium/results folder. An error might occur if a parameter set is used for which NE and/or SW computation has not been performed

## Computing Nash equilibria

1. Add karma_nash_equilibrium folder to Matlab path
2. Select high level parameters in load_parameters.m
	2.1. Multiple alpha values can be specified as a vector. It is recommended to specify in descending order
	2.2. Only one k_ave value can be specified
3. Select NE computation parameters in load_ne_parameters.m
4. Run karma_nash_equilibrium.m
5. Once computation is complete, move newly generated .mat file in karma_nash_equilibrium/results to correct subfolder

## Computing social welfare policies

1. Add karma_nash_equilibrium folder to Matlab path
2. Select high level parameters in load_parameters.m
	2.1 alpha is irrelevant
	2.2 Multiple k_ave values can be specified as a vector. It is recommended to specify in ascending order
3. Select SW computation parameters in load_ne_parameters.m
4. Run karma_social_welfare.m
5. Once computation is complete, move newly generated .mat file in karma_nash_equilibrium/results to correct subfolder

Note: It takes very long to set up optimization problem for large k_ave, before optimization solver starts output text. Be patient
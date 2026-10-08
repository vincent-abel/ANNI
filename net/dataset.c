/*
**	Dataset Code
*/
#include "dataset.h"

const double xor_inputs[XOR_SAMPLES * XOR_INPUTS] = {
	0.0, 0.0,
	0.0, 1.0,
	1.0, 0.0,
	1.0, 1.0
};

const double xor_targets[XOR_SAMPLES * XOR_OUTPUTS] = {
	0.0,
	1.0,
	1.0,
	0.0
};

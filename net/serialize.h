/*
**	Serialize Header
*/
#ifndef SERIALIZE_H__
#define SERIALIZE_H__

#include "layer.h"

int	network_save(network *net, const char *path);
network	*network_load(const char *path);

#endif

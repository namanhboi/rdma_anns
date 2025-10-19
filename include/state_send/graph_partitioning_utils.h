#pragma once

#include "defs.h"
#include "partitioning.h"

std::vector<std::vector<uint32_t>> get_partitions_from_adjgraph(std::vector<std::vector<int>> &adj,
                                    int num_partitions);



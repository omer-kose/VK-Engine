#pragma once

#include <vector>
#include "MaterialTypes.h"

namespace SK::Material
{
	struct MaterialRegistry
	{
		std::vector<Instance> instances;
	};

	uint32_t registerInstance(MaterialRegistry* materialRegistry, Instance&& instance);

	uint32_t clearMaterialRegistry(MaterialRegistry* materialRegistry);
}
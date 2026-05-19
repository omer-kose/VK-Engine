#pragma once

#include <vector>
#include "MaterialInfo.h"

namespace SK::Material
{
	struct MaterialRegistry
	{
		std::vector<Instance> instances;
	};

	uint32_t registerInstance(MaterialRegistry* materialRegistry, Instance&& instance);

	void clearMaterialRegistry(MaterialRegistry* materialRegistry);
}
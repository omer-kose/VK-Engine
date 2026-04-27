#pragma once

#include <glm/mat4x4.hpp>
#include <cstdint>

#include <AssetSystem/AssetTypes.h>
#include <MaterialSystem/MaterialTypes.h>

namespace SK::Renderer
{
	struct DrawPacket
	{
		glm::mat4x4 worldTransform;
		uint32_t meshIndex; // CPU and GPU asset registries have a 1-to-1 mapping. So, this index can directly be utilized for correct GPU mesh fetching.
		uint32_t startIndex;
		uint32_t indexCount;
		SK::Asset::MeshBounds bounds;
		uint32_t materialIndex = SK::Material::INVALID_MATERIAL;
	};
}
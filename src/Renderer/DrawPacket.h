#pragma once

#include <glm/mat4x4.hpp>
#include <string>
#include <cstdint>

#include <AssetSystem/AssetTypes.h>

namespace SK::Renderer
{
	struct DrawPacket
	{
		glm::mat4x4 worldTransform;
		uint32_t meshIndex; // CPU and GPU asset registries have a 1-to-1 mapping. So, this index can directly be utilized for correct GPU mesh fetching.
		uint32_t startIndex;
		uint32_t indexCount;
		SK::Asset::MeshBounds bounds;
	};
}
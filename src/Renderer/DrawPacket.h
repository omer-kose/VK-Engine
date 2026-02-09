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
		std::string meshName; // used to fetch GPU mesh
		uint32_t startIndex;
		uint32_t indexCount;
		SK::Asset::MeshBounds bounds;
	};
}
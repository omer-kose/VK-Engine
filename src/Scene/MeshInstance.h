#pragma once

#include <glm/mat4x4.hpp>
#include <cstdint>

namespace SK::Scene
{
	struct MeshInstance
	{
		glm::mat4 worldTransform;
		uint32_t meshIndex;
	};
}
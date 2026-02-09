#pragma once

#include <vector>
#include <string>
#include <glm/mat4x4.hpp>

namespace SK::Asset
{
	struct GLTFSceneNode
	{
		std::vector<int> children;
		glm::mat4 localTransform = glm::mat4(1.0f);
		int parent = -1;
		int meshIndex = -1; // index into related ImportedAsset's meshes
	};
	
	struct GLTFScene
	{
		std::vector<GLTFSceneNode> nodes;
		std::vector<int> rootNodes;
		std::string name;
	};
}
#pragma once

#include <vector>
#include <unordered_map>
#include <string>

#include <RendererBackend/vulkan/vk_types.h>

namespace SK::VkRendererBackend
{
	struct VkAssetRegistry
	{
		struct GPUMesh
		{
			GPUMeshBuffers meshBuffers;
			std::string name;
		};

		struct GPUTexture
		{
			AllocatedImage image;
			VkSampler sampler;
			std::string name;
		};

		std::vector<GPUMesh> meshes;
		std::vector<GPUTexture> textures;

		std::unordered_map<std::string, uint32_t> meshIndexByName;
		std::unordered_map<std::string, uint32_t> textureIndexByName;
	};
}
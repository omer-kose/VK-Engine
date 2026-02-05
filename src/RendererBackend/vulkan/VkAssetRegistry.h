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
			bool ownsImage; // false when fallback error image is used in the absence of image data
		};

		std::vector<GPUMesh> meshes;
		std::vector<GPUTexture> textures;

		std::unordered_map<std::string, uint32_t> meshIndexByName;
		std::unordered_map<std::string, uint32_t> textureIndexByName;
	};
}
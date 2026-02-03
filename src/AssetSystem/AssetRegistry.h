#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>

#include "Mesh.h"
#include "Texture.h"

namespace SK::VkRendererBackend
{
	struct State;
}

namespace SK::Asset
{
	struct AssetRegistry
	{
		std::vector<Mesh> meshes;
		std::vector<Texture> textures;
		// Samplers are not directly used, each texture holds a sampler handle. This container is used for clearing up the sampler created.
		std::vector<VkSampler> samplers;

		// String index mapping into the actual buffers
		std::unordered_map<std::string, uint32_t> meshIndexByName;
		std::unordered_map<std::string, uint32_t> textureIndexByName;
	};

	void clearAssetRegistry(SK::VkRendererBackend::State* vkRendererBackend, SK::Asset::AssetRegistry* assetRegistry);
};
#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>

#include "AssetTypes.h"
#include "ImportedAsset.h"

namespace SK::Asset
{
	struct AssetRegistry
	{
		std::vector<RawMesh> meshes;
		std::vector<RawTexture> textures;

		// String index mapping into the actual buffers
		std::unordered_map<std::string, uint32_t> meshIndexByName;
		std::unordered_map<std::string, uint32_t> textureIndexByName;
	};

	void registerImported(AssetRegistry* assetRegistry, ImportedAsset&& importedAsset);

	void discardCPUMeshData(AssetRegistry* assetRegistry);
	void discardCPUTextureData(AssetRegistry* assetRegistry);

	void clearAssetRegistry(SK::Asset::AssetRegistry* assetRegistry);
};
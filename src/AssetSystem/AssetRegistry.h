#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>

#include "AssetTypes.h"
#include <MaterialSystem/MaterialRegistry.h>
#include "ImportedAsset.h"

namespace SK::Asset
{
	struct AssetRegistry
	{
		std::vector<RawMesh> meshes;
		std::vector<RawTexture> textures;
		std::vector<GLTFScene> gltfScenes;

		// String index mapping into the actual buffers
		std::unordered_map<std::string, uint32_t> meshIndexByName;
		std::unordered_map<std::string, uint32_t> textureIndexByName;
		std::unordered_map<std::string, uint32_t> gltfSceneIndexByName;
	};

	void registerImported(AssetRegistry* assetRegistry, SK::Material::MaterialRegistry* materialRegistry, ImportedAsset&& importedAsset);

	void discardCPUMeshData(AssetRegistry* assetRegistry);
	void discardCPUTextureData(AssetRegistry* assetRegistry);

	void clearAssetRegistry(SK::Asset::AssetRegistry* assetRegistry);
};
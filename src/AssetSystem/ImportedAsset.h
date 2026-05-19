#pragma once

#include <vector>
#include <optional>

#include "AssetTypes.h"
#include <MaterialSystem/MaterialInfo.h>
#include "GLTFScene.h" // For optional GLTF-specific scene data

namespace SK::Asset
{
	// Temporary staging container used by importers. Registrating it to the Asset Registry will move its resources to the Asset Registry.
	struct ImportedAsset
	{
		std::vector<RawMesh> meshes;
		std::vector<RawTexture> textures;
		std::vector<SK::Material::Instance> materials;

		// Optional GLTF-specific scene data (if GLTF importer filled it. Other importers naturally won't touch this)
		std::optional<GLTFScene> gltfScene;
	};
}
#pragma once

#include <vector>
#include <optional>

#include "AssetTypes.h"
#include "GLTFScene.h" // For optional GLTF-specific scene data

namespace SK::Asset
{
	// Temporary staging container used by importers.
	struct ImportedAsset
	{
		std::vector<RawMesh> meshes;
		std::vector<RawTexture> textures;

		// Optional GLTF-specific scene data (if GLTF importer filled it. Other importers naturally won't touch this)
		std::optional<GLTFScene> gltfScene;
	};
}
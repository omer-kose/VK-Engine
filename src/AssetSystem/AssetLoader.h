#pragma once

#include <string_view>

namespace SK::VkRendererBackend
{
	struct State;
};

namespace SK::Asset
{ 
	struct AssetRegistry;
};

namespace SK::Asset
{
	// This only loads and registers the assets. It ignores the scene graph hierarchy. Should be used for only loading the asset data in the given gltf file not the whole scene.
	bool loadGLTFData(SK::VkRendererBackend::State* vkRendererBackend, SK::Asset::AssetRegistry* assetRegistry, std::string_view filePath);
};
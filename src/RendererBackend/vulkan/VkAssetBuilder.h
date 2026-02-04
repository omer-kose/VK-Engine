#pragma once

namespace SK::Asset
{
	struct AssetRegistry;
}

namespace SK::VkRendererBackend
{
	struct State;
	struct VkAssetRegistry;

	void buildGPUAssets(State* vkRendererBackend, SK::Asset::AssetRegistry* assetRegistry, VkAssetRegistry* vkAssetRegistry);
	void clearGPUAssets(State* vkRendererBackend, VkAssetRegistry* vkAssetRegistry);
}
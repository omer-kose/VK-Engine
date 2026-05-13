#pragma once

#include <RendererBackend/vulkan/VkAssetRegistry.h>
#include <RendererBackend/vulkan/VkMaterialRegistry.h>

namespace SK::Scene
{
	struct State;
};

namespace SK::VkRendererBackend
{
	struct State;

	struct VkSceneResources
	{
		VkAssetRegistry vkAssetRegistry;
		VkMaterialRegistry vkMaterialRegistry;

		// To make sure that assets are uploaded only once.
		bool isUploaded = false;
	};

	void uploadSceneResources(State* vkRendererBackend, SK::Scene::State* scene, VkSceneResources* outResources);

	void clearSceneResources(State* vkRendererBackend, VkSceneResources* resources);
}
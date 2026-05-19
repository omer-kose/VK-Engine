#include "VkSceneResources.h"

#include <cassert>

#include <Scene/Scene.h>

void SK::VkRendererBackend::uploadSceneResources(State* vkRendererBackend, SK::Scene::State* scene, VkSceneResources* outResources)
{
	assert(outResources->isUploaded == false);

	SK::VkRendererBackend::buildGPUAssets(vkRendererBackend, &scene->assetRegistry, &outResources->vkAssetRegistry);
	SK::VkRendererBackend::buildMaterialRegistry(vkRendererBackend, &scene->assetRegistry, &scene->materialRegistry, &outResources->vkAssetRegistry, &outResources->vkMaterialRegistry);

	outResources->isUploaded = true;
}

void SK::VkRendererBackend::clearSceneResources(State* vkRendererBackend, VkSceneResources* resources)
{
	if (resources && resources->isUploaded == true)
	{
		SK::VkRendererBackend::clearMaterialRegistry(vkRendererBackend, &resources->vkMaterialRegistry);
		SK::VkRendererBackend::clearGPUAssets(vkRendererBackend, &resources->vkAssetRegistry);

		resources->isUploaded = false;
	}
}

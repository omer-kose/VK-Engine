#pragma once

#include <vector>
#include <string_view>

#include <AssetSystem/AssetRegistry.h>
#include "MeshInstance.h"

namespace SK::Scene
{
    void buildMeshInstancesFromGLTFScene(const SK::Asset::AssetRegistry* assetRegistry, std::string_view sceneName, glm::mat4 sceneWorldTransform, std::vector<MeshInstance>& outInstances);
}
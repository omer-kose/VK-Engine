#include "GLTFInstanceBuilder.h"

#include <fmt/core.h>

static void traverseNode(const SK::Asset::GLTFScene& scene, int nodeIndex, const glm::mat4& parentTransform, std::vector<SK::Scene::MeshInstance>& outInstances)
{
    const auto& node = scene.nodes[nodeIndex];
    glm::mat4 world = parentTransform * node.localTransform;

    if(node.meshIndex >= 0)
    {
        SK::Scene::MeshInstance inst{};
        inst.meshIndex = static_cast<uint32_t>(node.meshIndex);
        inst.worldTransform = world;
        outInstances.push_back(inst);
    }

    for(int child : node.children)
    {
        traverseNode(scene, child, world, outInstances);
    }
}

void SK::Scene::buildMeshInstancesFromGLTFScene(const SK::Asset::AssetRegistry* assetRegistry, std::string_view sceneName, const glm::mat4& sceneWorldTransform, std::vector<MeshInstance>& outInstances)
{
    outInstances.clear();

    auto it = assetRegistry->gltfSceneIndexByName.find(std::string(sceneName));
    if(it == assetRegistry->gltfSceneIndexByName.end())
    {
        fmt::println("Cannot build instances. The given GLTF scene with the name {} doesn't exist.", sceneName);
        return;
    }

    const auto& scene = assetRegistry->gltfScenes[it->second];

    for(int root : scene.rootNodes)
    {
        traverseNode(scene, root, sceneWorldTransform, outInstances);
    }
}

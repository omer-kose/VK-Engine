#include <Application/Application.h>
#include <RendererBackend/vulkan/vk_renderer.h>
#include <UI/UI.h>
#include <Renderer/ForwardRenderer.h>

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

#include <thread>
#include <chrono>

#include <glm/gtx/transform.hpp>
// GPU Scene Data
GPUSceneData gpuSceneData;

// TODO: Asset System Test
#include <AssetSystem/AssetRegistry.h>
#include <AssetSystem/AssetImporter_GLTF.h>
#include <MaterialSystem/MaterialRegistry.h>
#include <RendererBackend/vulkan/VkAssetRegistry.h>
#include <RendererBackend/vulkan/VkMaterialRegistry.h>
#include <Scene/MeshInstance.h>
#include <Scene/GLTFInstanceBuilder.h>
#include <Renderer/DrawContext.h>
#include <Renderer/DrawPacketBuilder.h>

void updateSceneTemp(SK::VkRendererBackend::State* vkRendererBackend, Camera& camera)
{
    // TODO: Update timings are missing here but Engine Stats should be reconsidered too. Not sure if they should be in vkRendererBackend.
    camera.update();

    // TODO: Scene Data is directly set here. Not good!
    gpuSceneData.view = camera.getViewMatrix();
    // camera projection
    gpuSceneData.proj = glm::perspectiveRH_ZO(glm::radians(70.f), (float)vkRendererBackend->windowExtent.width / (float)vkRendererBackend->windowExtent.height, 0.1f, 10000.f);

    // invert the Y direction on projection matrix so that we are more similar
    // to opengl and gltf axis
    gpuSceneData.proj[1][1] *= -1;
    gpuSceneData.viewproj = gpuSceneData.proj * gpuSceneData.view;

    //some default lighting parameters
    gpuSceneData.ambientColor = glm::vec4(0.1f);
    gpuSceneData.sunlightColor = glm::vec4(1.0f);
    gpuSceneData.sunlightDirection = glm::vec4(0.0f, 1.0f, 0.5f, 1.0f);
}

int main(int argc, char* argv[])
{
	SK::Application::State application;
	SK::Application::init(&application, 1920, 1080);

	SK::VkRendererBackend::State vkRendererBackend;
	SK::VkRendererBackend::init(&vkRendererBackend, application.window, application.windowWidth, application.windowHeight);

    SK::UI::State ui;
    SK::UI::init(&ui, &vkRendererBackend);

    // Asset System
    SK::Asset::AssetRegistry assetRegistry;
    SK::Material::MaterialRegistry materialRegistry;
    SK::VkRendererBackend::VkAssetRegistry vkAssetRegistry;
    SK::VkRendererBackend::VkMaterialRegistry vkMaterialRegistry;
    // Load the structure scene
    SK::Asset::ImportedAsset structureScene;
    // Load and register the gltf scene
    // TODO: Infer the path from the name by providing the extension (glb or gltf)
    std::string gltfName = "structure";
    if (SK::Asset::importGLTF("../../assets/structure.glb", &structureScene))
    {
        if (structureScene.gltfScene.has_value())
        {
            SK::Asset::registerImported(&assetRegistry, &materialRegistry, std::move(structureScene));
            SK::VkRendererBackend::buildGPUAssets(&vkRendererBackend, &assetRegistry, &vkAssetRegistry);
            SK::VkRendererBackend::buildMaterialRegistry(&vkRendererBackend, &assetRegistry, &materialRegistry, &vkAssetRegistry, &vkMaterialRegistry);
            SK::Asset::discardCPUMeshData(&assetRegistry);
            SK::Asset::discardCPUTextureData(&assetRegistry);
        }
    }

    // Renderer frontends
    SK::ForwardRenderer::State forwardRenderer;
    // For descriptor layouts, vkRendererBackend and vkMaterialRegistry should be created before initializing renderers. 
    // NOTE: Just knowing the number of total textures for materials is enough to create a descriptor set layout for bindless resources. So, this is a soft constraint but still number of textures is need to be known.
    SK::ForwardRenderer::init(&forwardRenderer, &vkRendererBackend, &vkMaterialRegistry);
    // Draw context that renderer frontends will use
    SK::Renderer::DrawContext drawContext;

    // Instance the scene and fill in the draw context
    std::vector<SK::Scene::MeshInstance> meshInstances;
    SK::Scene::buildMeshInstancesFromGLTFScene(&assetRegistry, gltfName, glm::mat4(1.0f), meshInstances);
    // Create draw packets out of instances
    SK::Renderer::buildDrawPacketsFromMeshInstances(&assetRegistry, &materialRegistry, meshInstances, &drawContext);

    // main loop
    while(!application.shouldQuit)
    {
        // Begin frame time clock
        auto start = std::chrono::system_clock::now();

        SK::Application::handleSDLEvents(&application);

        // sleep if windows is minimized
        if(application.isMinimized)
        {
            // throttle the speed to avoid the endless spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // RendererBackend checks for a resize requirement every frame internally
        if(vkRendererBackend.windowResizeRequested)
        {
            SK::VkRendererBackend::handleWindowResize(&vkRendererBackend);
        }

        // --- UI FRAME BEGIN ---
        SK::UI::beginFrame();

        // --- PROGRAM SPECIFIC UI CODE ---
        ImGui::Begin("Stats");
        ImGui::Text("frametime %f ms", vkRendererBackend.stats.frameTime);
        ImGui::Text("geometry draw recording time %f ms", vkRendererBackend.stats.geometryDrawRecordTime);
        ImGui::Text("update time %f ms", vkRendererBackend.stats.sceneUpdateTime);
        ImGui::Text("triangles %i", vkRendererBackend.stats.triangleCount);
        ImGui::Text("draws %i", vkRendererBackend.stats.drawCallCount);
        ImGui::End();

        // --- UI FRAME END ---
        SK::UI::endFrame();

        updateSceneTemp(&vkRendererBackend, application.mainCamera);

        if(SK::VkRendererBackend::beginFrame(&vkRendererBackend))
        {
            SK::VkRendererBackend::updateSceneBuffer(&vkRendererBackend, gpuSceneData);
            SK::ForwardRenderer::draw(&forwardRenderer, &vkRendererBackend, &vkAssetRegistry, &vkMaterialRegistry, drawContext);
            SK::UI::draw(&vkRendererBackend);
            SK::VkRendererBackend::endFrame(&vkRendererBackend);
        }

        auto end = std::chrono::system_clock::now();
        // Convert to microseconds (integer), then come back to miliseconds
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        vkRendererBackend.stats.frameTime = elapsed.count() / 1000.0f;
    }

    // Make sure that GPU finished executing every command before shutting down the systems.
    vkDeviceWaitIdle(vkRendererBackend.device);

    SK::VkRendererBackend::clearMaterialRegistry(&vkRendererBackend, &vkMaterialRegistry);
    SK::VkRendererBackend::clearGPUAssets(&vkRendererBackend, &vkAssetRegistry);
    SK::Material::clearMaterialRegistry(&materialRegistry);
    SK::Asset::clearAssetRegistry(&assetRegistry);
    
    // Once everything is safe to delete shut the systems down.
    SK::ForwardRenderer::shutdown(&forwardRenderer, &vkRendererBackend);

    SK::UI::shutdown(&ui);

    SK::VkRendererBackend::shutdown(&vkRendererBackend);

	SK::Application::shutdown(&application);

	return 0;
}
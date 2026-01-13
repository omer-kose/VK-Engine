#include <Application/Application.h>
#include <Core/vk_renderer.h>
#include <UI/UI.h>

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

#include <thread>

// TODO: For now, scene loading and building the draw context is in the main function. They will be moved out
#include <Core/vk_loader.h>
#include <glm/gtx/transform.hpp>
// Loaded scenes
std::unordered_map<std::string, std::shared_ptr<LoadedGLTF>> loadedScenes;
// Draw Context
SK::VkRenderer::DrawContext drawContext;
// GPU Scene Data
GPUSceneData gpuSceneData;

void loadSceneData(SK::VkRenderer::Renderer* renderer)
{
    std::string structurePath = "../../assets/structure.glb";
    auto loadedStructureScene = loadGltf(renderer, structurePath);
    assert(loadedStructureScene.has_value());
    loadedScenes["structure"] = loadedStructureScene.value();
}

void loadScene(SK::VkRenderer::Renderer* renderer)
{
    loadSceneData(renderer);
}

void updateSceneTemp(SK::VkRenderer::Renderer* renderer, Camera& camera)
{
    // TODO: Update timings are missing here but Engine Stats should be reconsidered too. Not sure if they should be in renderer.

    camera.update();

    loadedScenes["structure"]->registerDraw(glm::mat4(1.0f), drawContext);

    // TODO: Scene Data is directly set here. Not good!
    gpuSceneData.view = camera.getViewMatrix();
    // camera projection
    gpuSceneData.proj = glm::perspectiveRH_ZO(glm::radians(70.f), (float)renderer->windowExtent.width / (float)renderer->windowExtent.height, 0.1f, 10000.f);

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
	SK::Application::Application application;
	SK::Application::init(&application, 1920, 1080);

	SK::VkRenderer::Renderer vkRenderer;
	SK::VkRenderer::init(&vkRenderer, application.window, application.windowWidth, application.windowHeight);

    SK::UI::UI ui;
    SK::UI::init(&ui, &vkRenderer);

    loadScene(&vkRenderer);

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

        // Renderer checks for a resize requirement every frame internally
        if(vkRenderer.resizeRequested)
        {
            SK::VkRenderer::m_resizeSwapchain(&vkRenderer);
        }

        // --- UI FRAME BEGIN ---
        SK::UI::beginFrame();

        // --- PROGRAM SPECIFIC UI CODE ---
        ImGui::Begin("Stats");
        ImGui::Text("frametime %f ms", vkRenderer.stats.frameTime);
        ImGui::Text("geometry draw recording time %f ms", vkRenderer.stats.geometryDrawRecordTime);
        ImGui::Text("update time %f ms", vkRenderer.stats.sceneUpdateTime);
        ImGui::Text("triangles %i", vkRenderer.stats.triangleCount);
        ImGui::Text("draws %i", vkRenderer.stats.drawCallCount);
        ImGui::End();

        // --- UI FRAME END ---
        SK::UI::endFrame();

        updateSceneTemp(&vkRenderer, application.mainCamera);

        SK::VkRenderer::draw(&vkRenderer, drawContext, gpuSceneData);
        
        // TODO: To be moved out to a proper place
        // After drawing clear out the DrawContext
        drawContext.clear();

        auto end = std::chrono::system_clock::now();
        // Convert to microseconds (integer), then come back to miliseconds
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        vkRenderer.stats.frameTime = elapsed.count() / 1000.0f;
    }

    // Make sure that GPU finished executing every command before shutting down the systems.
    vkDeviceWaitIdle(vkRenderer.device);

    // TODO: To be moved out to a proper place
    loadedScenes.clear();

    // Once everything is safe to delete shut the systems down.
    SK::UI::shutdown(&ui);

    SK::VkRenderer::shutdown(&vkRenderer);

	SK::Application::shutdown(&application);

	return 0;
}

#include <Application/Application.h>
#include <RendererBackend/vulkan/vk_renderer.h>
#include <UI/UI.h>

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

#include <thread>

// TODO: For now, scene loading and building the draw context is in the main function. They will be moved out
#include <RendererBackend/vulkan/vk_loader.h>
#include <glm/gtx/transform.hpp>
// Loaded scenes
std::unordered_map<std::string, std::shared_ptr<LoadedGLTF>> loadedScenes;
// Draw Context
SK::VkRendererBackend::DrawContext drawContext;
// GPU Scene Data
GPUSceneData gpuSceneData;

void loadSceneData(SK::VkRendererBackend::RendererBackend* vkRendererBackend)
{
    std::string structurePath = "../../assets/structure.glb";
    auto loadedStructureScene = loadGltf(vkRendererBackend, structurePath);
    assert(loadedStructureScene.has_value());
    loadedScenes["structure"] = loadedStructureScene.value();
}

void loadScene(SK::VkRendererBackend::RendererBackend* vkRendererBackend)
{
    loadSceneData(vkRendererBackend);
}

void updateSceneTemp(SK::VkRendererBackend::RendererBackend* vkRendererBackend, Camera& camera)
{
    // TODO: Update timings are missing here but Engine Stats should be reconsidered too. Not sure if they should be in vkRendererBackend.

    camera.update();

    loadedScenes["structure"]->registerDraw(glm::mat4(1.0f), drawContext);

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
	SK::Application::Application application;
	SK::Application::init(&application, 1920, 1080);

	SK::VkRendererBackend::RendererBackend vkRendererBackend;
	SK::VkRendererBackend::init(&vkRendererBackend, application.window, application.windowWidth, application.windowHeight);

    SK::UI::UI ui;
    SK::UI::init(&ui, &vkRendererBackend);

    loadScene(&vkRendererBackend);

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
        if(vkRendererBackend.resizeRequested)
        {
            SK::VkRendererBackend::m_resizeSwapchain(&vkRendererBackend);
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

        SK::VkRendererBackend::draw(&vkRendererBackend, drawContext, gpuSceneData);
        
        // TODO: To be moved out to a proper place
        // After drawing clear out the DrawContext
        drawContext.clear();

        auto end = std::chrono::system_clock::now();
        // Convert to microseconds (integer), then come back to miliseconds
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        vkRendererBackend.stats.frameTime = elapsed.count() / 1000.0f;
    }

    // Make sure that GPU finished executing every command before shutting down the systems.
    vkDeviceWaitIdle(vkRendererBackend.device);

    // TODO: To be moved out to a proper place
    loadedScenes.clear();

    // Once everything is safe to delete shut the systems down.
    SK::UI::shutdown(&ui);

    SK::VkRendererBackend::shutdown(&vkRendererBackend);

	SK::Application::shutdown(&application);

	return 0;
}

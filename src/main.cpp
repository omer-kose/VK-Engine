#include <Application/Application.h>
#include <RendererBackend/Vulkan/vk_renderer.h>
#include <RendererBackend/Vulkan/VkSceneResources.h>
#include <UI/UI.h>
#include <Scene/Scene.h>
#include <Renderer/GlobalGPUTypes.h>
#include <Renderer/ForwardRenderer.h>

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

#include <thread>
#include <chrono>

// Program specific Event Context
struct EventContext
{
    SK::Scene::State* scene = nullptr;
};

// Event callback that will be called by the Application Layer.
static void SDLEventCallback(const SDL_Event& event, void* eventContext)
{
    auto* context = static_cast<EventContext*>(eventContext);

    if (context && context->scene)
    {
        SDL_Event mutableEvent = event;
        context->scene->camera.processSDLEvent(mutableEvent);
    }

    SK::UI::processSDLEvents(event);
}

int main(int argc, char* argv[])
{
	SK::Application::State application;
	SK::Application::init(&application, 1920, 1080);

	SK::VkRendererBackend::State vkRendererBackend;
	SK::VkRendererBackend::init(&vkRendererBackend, application.window, application.windowWidth, application.windowHeight);

    SK::UI::State ui;
    SK::UI::init(&ui, &vkRendererBackend);

    SK::Scene::State scene;
    SK::Scene::setCameraProperties(&scene, glm::vec3(30.0f, 0.0f, -85.0f), 0.0f, 0.0f);
    SK::Scene::setProjectionProperties(&scene, 70.0f, 0.1f, 10000.0f);
    SK::Scene::setGlobalLightingProperties(&scene, glm::vec4(0.1f), glm::vec4(0.0f, 1.0f, 0.5f, 1.0f), glm::vec4(1.0f));
    const bool sceneLoaded = SK::Scene::loadGLTFScene(&scene, "../../assets/structure.glb");
    assert(sceneLoaded);

    SK::VkRendererBackend::VkSceneResources vkSceneResources;
    SK::VkRendererBackend::uploadSceneResources(&vkRendererBackend, &scene, &vkSceneResources);

    EventContext eventContext{};
    eventContext.scene = &scene;

    // Renderer frontends
    SK::ForwardRenderer::State forwardRenderer;
    // For descriptor layouts, vkRendererBackend and vkMaterialRegistry should be created before initializing renderers. 
    // NOTE: Just knowing the number of total textures for materials is enough to create a descriptor set layout for bindless resources. So, this is a soft constraint but still number of textures is need to be known.
    // TODO: Will write a RHI
    SK::ForwardRenderer::init(&forwardRenderer, &vkRendererBackend, &vkSceneResources.vkMaterialRegistry);

    // main loop
    while(!application.shouldQuit)
    {
        // Begin frame time clock
        auto start = std::chrono::system_clock::now();

        SK::Application::handleSDLEvents(&application, &SDLEventCallback, &eventContext);

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

        SK::Scene::updateCamera(&scene);
        SK::Scene::updateGPUSceneData(&scene, vkRendererBackend.windowExtent.width, vkRendererBackend.windowExtent.height);

        if(SK::VkRendererBackend::beginFrame(&vkRendererBackend))
        {
            SK::VkRendererBackend::updateSceneBuffer(&vkRendererBackend, scene.gpuSceneData);
            SK::ForwardRenderer::draw(&forwardRenderer, &vkRendererBackend, &vkSceneResources.vkAssetRegistry, &vkSceneResources.vkMaterialRegistry, scene.drawContext);
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

    SK::VkRendererBackend::clearSceneResources(&vkRendererBackend, &vkSceneResources);
    SK::Scene::clear(&scene);
    
    // Once everything is safe to delete shut the systems down.
    SK::ForwardRenderer::shutdown(&forwardRenderer, &vkRendererBackend);

    SK::UI::shutdown(&ui);

    SK::VkRendererBackend::shutdown(&vkRendererBackend);

	SK::Application::shutdown(&application);

	return 0;
}
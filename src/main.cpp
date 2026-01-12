#include <Application/Application.h>
#include <Core/vk_renderer.h>

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

#include <thread>

int main(int argc, char* argv[])
{
	SK::Application::Application application;
	SK::Application::init(&application, 1920, 1080);

	SK::VkRenderer::Renderer vkRenderer;
	SK::VkRenderer::init(&vkRenderer, application.window, application.windowWidth, application.windowHeight);

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

        // TODO: ImGui is handled here for now
        // ImGui new frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Stats");

        ImGui::Text("frametime %f ms", vkRenderer.stats.frameTime);
        ImGui::Text("geometry draw recording time %f ms", vkRenderer.stats.geometryDrawRecordTime);
        ImGui::Text("update time %f ms", vkRenderer.stats.sceneUpdateTime);
        ImGui::Text("triangles %i", vkRenderer.stats.triangleCount);
        ImGui::Text("draws %i", vkRenderer.stats.drawCallCount);
        ImGui::End();

        // Make ImGui calculate internal draw structures
        ImGui::Render();

        SK::VkRenderer::updateScene(&vkRenderer, &application.mainCamera);

        // TODO: Draw still calls to ImGui draw as ImGui requires command buffer that is being used that frame as well
        SK::VkRenderer::draw(&vkRenderer);

        auto end = std::chrono::system_clock::now();
        // Convert to microseconds (integer), then come back to miliseconds
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        vkRenderer.stats.frameTime = elapsed.count() / 1000.0f;
    }

	SK::VkRenderer::shutdown(&vkRenderer);

	SK::Application::shutdown(&application);

	return 0;
}

#include <Application/Application.h>
#include <Core/vk_renderer.h>

int main(int argc, char* argv[])
{
	SK::Application::Application application;
	// Window width and height are defaulted for now
	SK::Application::init(&application);

	SK::VkRenderer::Renderer vkRenderer;

	SK::VkRenderer::init(&vkRenderer, application.window, application.windowWidth, application.windowHeight);

	SK::VkRenderer::run(&vkRenderer);

	SK::VkRenderer::cleanup(&vkRenderer);

	return 0;
}

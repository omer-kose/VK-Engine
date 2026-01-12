#include <Application/Application.h>
#include <Core/vk_renderer.h>

int main(int argc, char* argv[])
{
	SK::Application::Application application;
	SK::Application::init(&application, 1920, 1080);

	SK::VkRenderer::Renderer vkRenderer;
	SK::VkRenderer::init(&vkRenderer, application.window, application.windowWidth, application.windowHeight);
	SK::VkRenderer::run(&vkRenderer);
	SK::VkRenderer::cleanup(&vkRenderer);

	return 0;
}

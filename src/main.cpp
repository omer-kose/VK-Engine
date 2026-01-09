#include <Core/vk_renderer.h>

int main(int argc, char* argv[])
{
	SK::VkRenderer::Renderer vkRenderer;

	SK::VkRenderer::init(&vkRenderer);

	SK::VkRenderer::run(&vkRenderer);

	SK::VkRenderer::cleanup(&vkRenderer);

	return 0;
}

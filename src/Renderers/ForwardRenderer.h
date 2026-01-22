#pragma once
#include <RendererBackend/vulkan/vk_types.h>

// Forward declare with the namespace
namespace SK::VkRendererBackend
{
	struct RendererBackend;
	struct DrawContext;
};

namespace SK::ForwardRenderer
{
	struct ForwardRenderer
	{
		VkPipelineLayout pipelineLayout; // both transparent and opaque objects use the same pipeline layout
		VkPipeline opaquePipeline;
		VkPipeline transparentPipeline;
	};

	void init(ForwardRenderer* forwardRenderer, SK::VkRendererBackend::RendererBackend* vkRendererBackend);
	void draw(ForwardRenderer* forwardRenderer, SK::VkRendererBackend::RendererBackend* vkRendererBackend, const SK::VkRendererBackend::DrawContext& ctx);
	void shutdown(ForwardRenderer* forwardRenderer, SK::VkRendererBackend::RendererBackend* vkRendererBackend);
};
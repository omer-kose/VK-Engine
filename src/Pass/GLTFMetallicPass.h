#pragma once
#include <RendererBackend/vulkan/vk_types.h>

// Forward declare with the namespace
namespace SK::VkRendererBackend
{
	struct RendererBackend;
	struct RenderObject;
	struct DrawContext;
};

class GLTFMetallicPass
{
public:
	static void Init(SK::VkRendererBackend::RendererBackend* vkRendererBackend);
	static void Execute(SK::VkRendererBackend::RendererBackend* vkRendererBackend, VkCommandBuffer& cmd, const SK::VkRendererBackend::DrawContext& ctx);
	static void Update();
	static void ClearResources(SK::VkRendererBackend::RendererBackend* vkRendererBackend);
private:
	static VkPipeline OpaquePipeline;
	static VkPipeline TransparentPipeline;
	static VkPipelineLayout PipelineLayout; // both transparent and opaque objects use the same pipeline layout
};
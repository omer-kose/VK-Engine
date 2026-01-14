#pragma once
#include <RendererBackend/vulkan/vk_types.h>

// Forward declare with the namespace
namespace SK::VkRendererBackend
{
	struct Renderer;
	struct RenderObject;
	struct DrawContext;
};

class GLTFMetallicPass
{
public:
	static void Init(SK::VkRendererBackend::Renderer* renderer);
	static void Execute(SK::VkRendererBackend::Renderer* renderer, VkCommandBuffer& cmd, const SK::VkRendererBackend::DrawContext& ctx);
	static void Update();
	static void ClearResources(SK::VkRendererBackend::Renderer* renderer);
private:
	static VkPipeline OpaquePipeline;
	static VkPipeline TransparentPipeline;
	static VkPipelineLayout PipelineLayout; // both transparent and opaque objects use the same pipeline layout
};
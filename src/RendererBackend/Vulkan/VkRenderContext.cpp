#include "VkRenderContext.h"

#include <Renderer/RenderContext.h>

#include <RendererBackend/Vulkan/VkAssetRegistry.h>
#include <RendererBackend/Vulkan/VkInitializers.h>
#include <RendererBackend/Vulkan/VkMaterialRegistry.h>
#include <RendererBackend/Vulkan/VkPipelines.h>
#include <RendererBackend/Vulkan/VkRendererBackend.h>
#include <RendererBackend/Vulkan/VkSceneResources.h>

#include <fmt/core.h>

#include <functional>
#include <string>

static constexpr uint32_t SCENE_RESOURCE_SET_SLOT = 0;
static constexpr uint32_t MATERIAL_RESOURCE_SET_SLOT = 1;

static SK::VkRendererBackend::VkRenderContext* fetchVkRenderContext(SK::Renderer::RenderContext* renderContext)
{
	return static_cast<SK::VkRendererBackend::VkRenderContext*>(renderContext->backend);
}

static VkShaderStageFlags toVkShaderStageFlags(SK::Renderer::ShaderStageFlags stages)
{
	VkShaderStageFlags vkStages = 0;

	if ((stages & SK::Renderer::ShaderStageFlagBits::VertexShader) != 0)
	{
		vkStages |= VK_SHADER_STAGE_VERTEX_BIT;
	}

	if ((stages & SK::Renderer::ShaderStageFlagBits::FragmentShader) != 0)
	{
		vkStages |= VK_SHADER_STAGE_FRAGMENT_BIT;
	}

	if ((stages & SK::Renderer::ShaderStageFlagBits::ComputeShader) != 0)
	{
		vkStages |= VK_SHADER_STAGE_COMPUTE_BIT;
	}

	return vkStages;
}

static VkPrimitiveTopology toVkPrimitiveTopology(SK::Renderer::PrimitiveTopology topology)
{
	switch (topology)
	{
	case SK::Renderer::PrimitiveTopology::TriangleList:
		return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	case SK::Renderer::PrimitiveTopology::LineList:
		return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
	case SK::Renderer::PrimitiveTopology::PointList:
		return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
	default:
		return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	}
}

static VkPolygonMode toVkPolygonMode(SK::Renderer::PolygonMode polygonMode)
{
	switch (polygonMode)
	{
	case SK::Renderer::PolygonMode::Fill:
		return VK_POLYGON_MODE_FILL;
	case SK::Renderer::PolygonMode::Line:
		return VK_POLYGON_MODE_LINE;
	default:
		return VK_POLYGON_MODE_FILL;
	}
}

static VkCullModeFlags toVkCullModeFlags(SK::Renderer::CullMode cullMode)
{
	switch (cullMode)
	{
	case SK::Renderer::CullMode::None:
		return VK_CULL_MODE_NONE;
	case SK::Renderer::CullMode::Front:
		return VK_CULL_MODE_FRONT_BIT;
	case SK::Renderer::CullMode::Back:
		return VK_CULL_MODE_BACK_BIT;
	default:
		return VK_CULL_MODE_NONE;
	}
}

static VkFrontFace toVkFrontFace(SK::Renderer::FrontFace frontFace)
{
	switch (frontFace)
	{
	case SK::Renderer::FrontFace::Clockwise:
		return VK_FRONT_FACE_CLOCKWISE;
	case SK::Renderer::FrontFace::CounterClockwise:
		return VK_FRONT_FACE_COUNTER_CLOCKWISE;
	default:
		return VK_FRONT_FACE_COUNTER_CLOCKWISE;
	}
}

static VkCompareOp toVkCompareOp(SK::Renderer::CompareOp compareOp)
{
	switch (compareOp)
	{
	case SK::Renderer::CompareOp::Never:
		return VK_COMPARE_OP_NEVER;
	case SK::Renderer::CompareOp::Less:
		return VK_COMPARE_OP_LESS;
	case SK::Renderer::CompareOp::LessEqual:
		return VK_COMPARE_OP_LESS_OR_EQUAL;
	case SK::Renderer::CompareOp::Equal:
		return VK_COMPARE_OP_EQUAL;
	case SK::Renderer::CompareOp::GreaterEqual:
		return VK_COMPARE_OP_GREATER_OR_EQUAL;
	case SK::Renderer::CompareOp::Greater:
		return VK_COMPARE_OP_GREATER;
	case SK::Renderer::CompareOp::Always:
		return VK_COMPARE_OP_ALWAYS;
	default:
		return VK_COMPARE_OP_LESS_OR_EQUAL;
	}
}

static VkIndexType toVkIndexType(SK::Renderer::IndexType indexType)
{
	switch (indexType)
	{
	case SK::Renderer::IndexType::Uint16:
		return VK_INDEX_TYPE_UINT16;
	case SK::Renderer::IndexType::Uint32:
		return VK_INDEX_TYPE_UINT32;
	default:
		return VK_INDEX_TYPE_UINT32;
	}
}

static VkPipelineBindPoint toVkPipelineBindPoint(SK::Renderer::PipelineKind kind)
{
	switch (kind)
	{
	case SK::Renderer::PipelineKind::Graphics:
		return VK_PIPELINE_BIND_POINT_GRAPHICS;
	case SK::Renderer::PipelineKind::Compute:
		return VK_PIPELINE_BIND_POINT_COMPUTE;
	default:
		return VK_PIPELINE_BIND_POINT_GRAPHICS;
	}
}

static void hashCombine(size_t* seed, size_t value)
{
	*seed ^= value + 0x9e3779b9 + (*seed << 6) + (*seed >> 2);
}

static size_t hashString(const char* str)
{
	return std::hash<std::string>{}(str ? str : "");
}

static void hashCustomResourceSets(size_t* hash, const std::vector<SK::Renderer::PipelineResourceSet>& customResourceSets)
{
	std::hash<uint64_t> integerHasher;

	hashCombine(hash, integerHasher(static_cast<uint64_t>(customResourceSets.size())));

	for (const SK::Renderer::PipelineResourceSet& customSet : customResourceSets)
	{
		hashCombine(hash, integerHasher(customSet.slot));
		hashCombine(hash, integerHasher(customSet.set.id));
	}
}

static size_t hashGraphicsPipelineDesc(const SK::Renderer::GraphicsPipelineDesc& desc)
{
	size_t hash = 0;

	std::hash<uint64_t> integerHasher;

	hashCombine(&hash, integerHasher(static_cast<uint64_t>(SK::Renderer::PipelineKind::Graphics)));
	hashCombine(&hash, hashString(desc.vertexShaderPath));
	hashCombine(&hash, hashString(desc.fragmentShaderPath));

	hashCombine(&hash, integerHasher(static_cast<uint64_t>(desc.topology)));
	hashCombine(&hash, integerHasher(static_cast<uint64_t>(desc.polygonMode)));
	hashCombine(&hash, integerHasher(static_cast<uint64_t>(desc.cullMode)));
	hashCombine(&hash, integerHasher(static_cast<uint64_t>(desc.frontFace)));

	hashCombine(&hash, integerHasher(desc.depthTest ? 1 : 0));
	hashCombine(&hash, integerHasher(desc.depthWrite ? 1 : 0));
	hashCombine(&hash, integerHasher(static_cast<uint64_t>(desc.depthCompare)));

	hashCombine(&hash, integerHasher(desc.blending ? 1 : 0));

	hashCombine(&hash, integerHasher(desc.pushConstantSize));
	hashCombine(&hash, integerHasher(desc.pushConstantStages));

	hashCombine(&hash, integerHasher(desc.usesSceneResources ? 1 : 0));
	hashCombine(&hash, integerHasher(desc.usesMaterialResources ? 1 : 0));
	hashCustomResourceSets(&hash, desc.customResourceSets);

	return hash;
}

static size_t hashComputePipelineDesc(const SK::Renderer::ComputePipelineDesc& desc)
{
	size_t hash = 0;

	std::hash<uint64_t> integerHasher;

	hashCombine(&hash, integerHasher(static_cast<uint64_t>(SK::Renderer::PipelineKind::Compute)));
	hashCombine(&hash, hashString(desc.computeShaderPath));

	hashCombine(&hash, integerHasher(desc.pushConstantSize));
	hashCombine(&hash, integerHasher(desc.pushConstantStages));

	hashCombine(&hash, integerHasher(desc.usesSceneResources ? 1 : 0));
	hashCombine(&hash, integerHasher(desc.usesMaterialResources ? 1 : 0));
	hashCustomResourceSets(&hash, desc.customResourceSets);

	return hash;
}

static SK::VkRendererBackend::PipelineLayoutKey buildPipelineLayoutKey(
	SK::VkRendererBackend::VkRenderContext* vkRenderContext,
	bool usesSceneResources,
	bool usesMaterialResources,
	const std::vector<SK::Renderer::PipelineResourceSet>& customResourceSets,
	uint32_t pushConstantSize,
	SK::Renderer::ShaderStageFlags pushConstantStages)
{
	SK::VkRendererBackend::PipelineLayoutKey layoutKey{};

	uint32_t numResourceSets = 0;
	numResourceSets += (usesSceneResources == true);
	numResourceSets += (usesMaterialResources == true);
	numResourceSets += customResourceSets.size();

	layoutKey.setLayouts.resize(numResourceSets, VK_NULL_HANDLE);

	if (usesSceneResources)
	{
		layoutKey.setLayouts[SCENE_RESOURCE_SET_SLOT] = vkRenderContext->vkRendererBackend->gpuSceneDataDescriptorLayout;
	}

	if (usesMaterialResources)
	{
		layoutKey.setLayouts[MATERIAL_RESOURCE_SET_SLOT] = vkRenderContext->sceneResources->vkMaterialRegistry.resourceDescriptorSetLayout;
	}

	for (const SK::Renderer::PipelineResourceSet& customSet : customResourceSets)
	{
		layoutKey.setLayouts[customSet.slot] = vkRenderContext->customResourceRecords[customSet.set.id].layout;
	}

	if (pushConstantSize > 0)
	{
		VkPushConstantRange pushConstantRange{};
		pushConstantRange.offset = 0;
		pushConstantRange.size = pushConstantSize;
		pushConstantRange.stageFlags = toVkShaderStageFlags(pushConstantStages);

		layoutKey.pushConstantRanges.push_back(pushConstantRange);
	}

	return layoutKey;
}

static SK::Renderer::PipelineHandle vkGetGraphicsPipeline(SK::Renderer::RenderContext* renderContext, const SK::Renderer::GraphicsPipelineDesc& desc)
{
	SK::VkRendererBackend::VkRenderContext* vkRenderContext = fetchVkRenderContext(renderContext);
	SK::VkRendererBackend::State* vkRendererBackend = vkRenderContext->vkRendererBackend;
	SK::VkRendererBackend::VkSceneResources* sceneResources = vkRenderContext->sceneResources;

	const size_t descHash = hashGraphicsPipelineDesc(desc);

	auto existing = vkRenderContext->pipelineIndexByHash.find(descHash);
	if (existing != vkRenderContext->pipelineIndexByHash.end())
	{
		return SK::Renderer::PipelineHandle{ existing->second };
	}

	VkShaderModule vertexShader = SK::VkRendererBackend::getOrLoadShader(vkRendererBackend, desc.vertexShaderPath);
	if (vertexShader == VK_NULL_HANDLE)
	{
		fmt::println("Failed to load vertex shader: {}", desc.vertexShaderPath ? desc.vertexShaderPath : "<null>");
		return SK::Renderer::PipelineHandle{};
	}

	VkShaderModule fragmentShader = SK::VkRendererBackend::getOrLoadShader(vkRendererBackend, desc.fragmentShaderPath);
	if (fragmentShader == VK_NULL_HANDLE)
	{
		fmt::println("Failed to load fragment shader: {}", desc.fragmentShaderPath ? desc.fragmentShaderPath : "<null>");
		return SK::Renderer::PipelineHandle{};
	}

	SK::VkRendererBackend::PipelineLayoutKey layoutKey = buildPipelineLayoutKey(
		vkRenderContext,
		desc.usesSceneResources,
		desc.usesMaterialResources,
		desc.customResourceSets,
		desc.pushConstantSize,
		desc.pushConstantStages
	);

	VkPipelineLayout pipelineLayout = SK::VkRendererBackend::getOrCreatePipelineLayout(vkRendererBackend, layoutKey);

	const size_t vertexShaderHash = hashString(desc.vertexShaderPath);
	const size_t fragmentShaderHash = hashString(desc.fragmentShaderPath);

	SK::VkRendererBackend::PipelineKey pipelineKey{};
	pipelineKey.vertShader = vertexShaderHash;
	pipelineKey.fragShader = fragmentShaderHash;
	pipelineKey.topology = toVkPrimitiveTopology(desc.topology);
	pipelineKey.polygonMode = toVkPolygonMode(desc.polygonMode);
	pipelineKey.cullMode = toVkCullModeFlags(desc.cullMode);
	pipelineKey.frontFace = toVkFrontFace(desc.frontFace);
	pipelineKey.depthTest = desc.depthTest;
	pipelineKey.depthWrite = desc.depthWrite;
	pipelineKey.depthCompare = toVkCompareOp(desc.depthCompare);
	pipelineKey.blending = desc.blending;
	pipelineKey.colorFormat = vkRendererBackend->drawImage.imageFormat;
	pipelineKey.depthFormat = vkRendererBackend->depthImage.imageFormat;
	pipelineKey.layout = pipelineLayout;

	VkPipeline pipeline = SK::VkRendererBackend::getOrCreatePipeline(vkRendererBackend, pipelineKey);

	SK::VkRendererBackend::PipelineRecord record{};
	record.kind = SK::Renderer::PipelineKind::Graphics;
	record.pipeline = pipeline;
	record.layout = pipelineLayout;

	const uint64_t pipelineIndex = static_cast<uint64_t>(vkRenderContext->pipelines.size());
	vkRenderContext->pipelines.push_back(record);
	vkRenderContext->pipelineIndexByHash[descHash] = pipelineIndex;

	return SK::Renderer::PipelineHandle{ pipelineIndex };
}

// TODO: To be implemented
static SK::Renderer::PipelineHandle vkGetComputePipeline(SK::Renderer::RenderContext* renderContext, const SK::Renderer::ComputePipelineDesc& desc)
{
	return SK::Renderer::PipelineHandle{ SK::Renderer::INVALID_HANDLE };
}

static SK::Renderer::BufferDeviceAddress vkGetVertexBufferDeviceAddress(SK::Renderer::RenderContext* renderContext, size_t meshIndex)
{
	SK::VkRendererBackend::VkRenderContext* vkRenderContext = fetchVkRenderContext(renderContext);
	SK::VkRendererBackend::State* vkRendererBackend = vkRenderContext->vkRendererBackend;
	SK::VkRendererBackend::VkSceneResources* sceneResources = vkRenderContext->sceneResources;

	if (meshIndex >= sceneResources->vkAssetRegistry.meshes.size())
	{
		fmt::println("Tried to get the vertex buffer address of an invalid mesh: {}", meshIndex);
		return static_cast<SK::Renderer::BufferDeviceAddress>(SK::Renderer::INVALID_HANDLE);
	}

	const SK::VkRendererBackend::VkAssetRegistry::GPUMesh& mesh = sceneResources->vkAssetRegistry.meshes[meshIndex];
	return static_cast<SK::Renderer::BufferDeviceAddress>(mesh.meshBuffers.vertexBufferAddress);
}

static void vkBeginMainRendering(SK::Renderer::RenderContext* renderContext)
{
	SK::VkRendererBackend::VkRenderContext* vkRenderContext = fetchVkRenderContext(renderContext);
	SK::VkRendererBackend::State* vkRendererBackend = vkRenderContext->vkRendererBackend;

	VkCommandBuffer cmd = vkRendererBackend->currentCmdBuffer;

	VkRenderingAttachmentInfo colorAttachment = SK::VkInit::attachment_info(
		vkRendererBackend->drawImage.imageView,
		&vkRendererBackend->colorAttachmentClearValue,
		VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
	);

	VkRenderingAttachmentInfo depthAttachment = SK::VkInit::depth_attachment_info(
		vkRendererBackend->depthImage.imageView,
		VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
	);

	VkRenderingInfo renderInfo = SK::VkInit::rendering_info(
		vkRendererBackend->drawExtent,
		&colorAttachment,
		&depthAttachment
	);

	vkCmdBeginRendering(cmd, &renderInfo);
}

static void vkEndRendering(SK::Renderer::RenderContext* renderContext)
{
	SK::VkRendererBackend::VkRenderContext* vkRenderContext = fetchVkRenderContext(renderContext);
	SK::VkRendererBackend::State* vkRendererBackend = vkRenderContext->vkRendererBackend;

	vkCmdEndRendering(vkRendererBackend->currentCmdBuffer);
}

static void vkBindPipeline(SK::Renderer::RenderContext* renderContext, SK::Renderer::PipelineHandle pipeline)
{
	SK::VkRendererBackend::VkRenderContext* vkRenderContext = fetchVkRenderContext(renderContext);
	SK::VkRendererBackend::State* vkRendererBackend = vkRenderContext->vkRendererBackend;

	if (pipeline.id >= vkRenderContext->pipelines.size())
	{
		fmt::println("Invalid pipeline handle: {}", pipeline.id);
		return;
	}

	const SK::VkRendererBackend::PipelineRecord& record = vkRenderContext->pipelines[static_cast<size_t>(pipeline.id)];

	vkRenderContext->currentPipelineKind = record.kind;
	vkRenderContext->currentPipelineLayout = record.layout;

	vkCmdBindPipeline(vkRendererBackend->currentCmdBuffer, toVkPipelineBindPoint(record.kind), record.pipeline);

	if (record.kind == SK::Renderer::PipelineKind::Graphics)
	{
		// Automatically setting the dynamic state to default settings. For future passes requiring, custom viewport and scissor, I might need to take this out and provide functionality to manually set.
		SK::VkRendererBackend::setViewport(vkRendererBackend, vkRendererBackend->currentCmdBuffer);
		SK::VkRendererBackend::setScissor(vkRendererBackend, vkRendererBackend->currentCmdBuffer);
	}
}

static void vkBindSceneResources(SK::Renderer::RenderContext* renderContext)
{
	SK::VkRendererBackend::VkRenderContext* vkRenderContext = fetchVkRenderContext(renderContext);
	SK::VkRendererBackend::State* vkRendererBackend = vkRenderContext->vkRendererBackend;

	VkDescriptorSet sceneDescriptorSet = SK::VkRendererBackend::fetchCurrentSceneBufferDescriptorSet(vkRendererBackend);

	vkCmdBindDescriptorSets(
		vkRendererBackend->currentCmdBuffer,
		toVkPipelineBindPoint(vkRenderContext->currentPipelineKind),
		vkRenderContext->currentPipelineLayout,
		SCENE_RESOURCE_SET_SLOT,
		1,
		&sceneDescriptorSet,
		0,
		nullptr
	);
}

static void vkBindMaterialResources(SK::Renderer::RenderContext* renderContext)
{
	SK::VkRendererBackend::VkRenderContext* vkRenderContext = fetchVkRenderContext(renderContext);
	SK::VkRendererBackend::State* vkRendererBackend = vkRenderContext->vkRendererBackend;
	SK::VkRendererBackend::VkSceneResources* sceneResources = vkRenderContext->sceneResources;

	vkCmdBindDescriptorSets(
		vkRendererBackend->currentCmdBuffer,
		toVkPipelineBindPoint(vkRenderContext->currentPipelineKind),
		vkRenderContext->currentPipelineLayout,
		MATERIAL_RESOURCE_SET_SLOT,
		1,
		&sceneResources->vkMaterialRegistry.resourceDescriptorSet,
		0,
		nullptr
	);
}

static void vkBindResourceSet(SK::Renderer::RenderContext* ctx, uint32_t slot, SK::Renderer::ResourceSetHandle set)
{
	SK::VkRendererBackend::VkRenderContext* vkRenderContext = fetchVkRenderContext(ctx);
	SK::VkRendererBackend::State* vkRendererBackend = vkRenderContext->vkRendererBackend;

	const SK::VkRendererBackend::ResourceRecord record = vkRenderContext->customResourceRecords[set.id];

	vkCmdBindDescriptorSets(
		vkRendererBackend->currentCmdBuffer,
		toVkPipelineBindPoint(vkRenderContext->currentPipelineKind),
		vkRenderContext->currentPipelineLayout,
		slot,
		1,
		&record.set,
		0,
		nullptr
	);
}

static void vkPushConstants(SK::Renderer::RenderContext* renderContext, SK::Renderer::ShaderStageFlags stages, uint32_t offset, uint32_t size, const void* data)
{
	SK::VkRendererBackend::VkRenderContext* vkRenderContext = fetchVkRenderContext(renderContext);
	SK::VkRendererBackend::State* vkRendererBackend = vkRenderContext->vkRendererBackend;

	vkCmdPushConstants(
		vkRendererBackend->currentCmdBuffer,
		vkRenderContext->currentPipelineLayout,
		toVkShaderStageFlags(stages),
		offset,
		size,
		data
	);
}

static void vkBindIndexBuffer(SK::Renderer::RenderContext* renderContext, size_t meshIndex, SK::Renderer::IndexType indexType)
{
	SK::VkRendererBackend::VkRenderContext* vkRenderContext = fetchVkRenderContext(renderContext);
	SK::VkRendererBackend::State* vkRendererBackend = vkRenderContext->vkRendererBackend;
	SK::VkRendererBackend::VkSceneResources* sceneResources = vkRenderContext->sceneResources;

	if (meshIndex >= sceneResources->vkAssetRegistry.meshes.size())
	{
		fmt::println("Tried to bind the index buffer of an invalid mesh index: {}", meshIndex);
		return;
	}

	const SK::VkRendererBackend::VkAssetRegistry::GPUMesh& mesh = sceneResources->vkAssetRegistry.meshes[meshIndex];

	vkCmdBindIndexBuffer(
		vkRendererBackend->currentCmdBuffer,
		mesh.meshBuffers.indexBuffer.buffer,
		0,
		toVkIndexType(indexType)
	);
}

static void vkDrawIndexed(SK::Renderer::RenderContext* renderContext, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance)
{
	SK::VkRendererBackend::VkRenderContext* vkRenderContext = fetchVkRenderContext(renderContext);
	SK::VkRendererBackend::State* vkRendererBackend = vkRenderContext->vkRendererBackend;

	vkCmdDrawIndexed(
		vkRendererBackend->currentCmdBuffer,
		indexCount,
		instanceCount,
		firstIndex,
		vertexOffset,
		firstInstance
	);
}

static void vkDispatch(SK::Renderer::RenderContext* renderContext, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
{
	SK::VkRendererBackend::VkRenderContext* vkRenderContext = fetchVkRenderContext(renderContext);
	SK::VkRendererBackend::State* vkRendererBackend = vkRenderContext->vkRendererBackend;

	vkCmdDispatch(
		vkRendererBackend->currentCmdBuffer,
		groupCountX,
		groupCountY,
		groupCountZ
	);
}

void SK::VkRendererBackend::initVkRenderContext(VkRenderContext* vkRenderContext, State* vkRendererBackend, VkSceneResources* vkSceneResources)
{
	vkRenderContext->vkRendererBackend = vkRendererBackend;
	vkRenderContext->sceneResources = vkSceneResources;
	vkRenderContext->pipelines.clear();
	vkRenderContext->pipelineIndexByHash.clear();
	vkRenderContext->customResourceRecords.clear();
	vkRenderContext->customResourceLayoutByHash.clear();
	vkRenderContext->currentPipelineKind = SK::Renderer::PipelineKind::Graphics;
	vkRenderContext->currentPipelineLayout = VK_NULL_HANDLE;
}

SK::Renderer::RenderContext SK::VkRendererBackend::makeRenderContext(VkRenderContext* vkRenderContext)
{
	static const SK::Renderer::RenderContextAPI api =
	{
		.getGraphicsPipeline = vkGetGraphicsPipeline,
		.getComputePipeline = vkGetComputePipeline,
		.getVertexBufferDeviceAddress = vkGetVertexBufferDeviceAddress,
		.beginMainRendering = vkBeginMainRendering,
		.endRendering = vkEndRendering,
		.bindPipeline = vkBindPipeline,
		.bindSceneResources = vkBindSceneResources,
		.bindMaterialResources = vkBindMaterialResources,
		.bindResourceSet = vkBindResourceSet,
		.pushConstants = vkPushConstants,
		.bindIndexBuffer = vkBindIndexBuffer,
		.drawIndexed = vkDrawIndexed,
		.dispatch = vkDispatch,
	};

	SK::Renderer::RenderContext renderContext{};
	renderContext.backend = vkRenderContext;
	renderContext.api = &api;

	return renderContext;
}

void SK::VkRendererBackend::clearVkRenderContext(VkRenderContext* vkRenderContext)
{
	if (vkRenderContext->vkRendererBackend != nullptr)
	{
		for (ResourceRecord& record : vkRenderContext->customResourceRecords)
		{
			if (record.ownsLayout && record.layout != VK_NULL_HANDLE)
			{
				vkDestroyDescriptorSetLayout(vkRenderContext->vkRendererBackend->device, record.layout, nullptr);
				record.layout = VK_NULL_HANDLE;
			}

			// Descriptor sets are owned by descriptor pools.
			record.set = VK_NULL_HANDLE;
		}
	}

	vkRenderContext->vkRendererBackend = nullptr;
	vkRenderContext->sceneResources = nullptr;
	vkRenderContext->pipelines.clear();
	vkRenderContext->pipelineIndexByHash.clear();
	vkRenderContext->customResourceRecords.clear();
	vkRenderContext->customResourceLayoutByHash.clear();
	vkRenderContext->currentPipelineKind = SK::Renderer::PipelineKind::Graphics;
	vkRenderContext->currentPipelineLayout = VK_NULL_HANDLE;
}

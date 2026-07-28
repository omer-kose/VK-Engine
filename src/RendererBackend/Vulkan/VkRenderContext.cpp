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
		case SK::Renderer::CompareOp::Never:          return VK_COMPARE_OP_NEVER;
		case SK::Renderer::CompareOp::Less:           return VK_COMPARE_OP_LESS;
		case SK::Renderer::CompareOp::Equal:          return VK_COMPARE_OP_EQUAL;
		case SK::Renderer::CompareOp::LessOrEqual:    return VK_COMPARE_OP_LESS_OR_EQUAL;
		case SK::Renderer::CompareOp::Greater:        return VK_COMPARE_OP_GREATER;
		case SK::Renderer::CompareOp::NotEqual:       return VK_COMPARE_OP_NOT_EQUAL;
		case SK::Renderer::CompareOp::GreaterOrEqual: return VK_COMPARE_OP_GREATER_OR_EQUAL;
		case SK::Renderer::CompareOp::Always:         return VK_COMPARE_OP_ALWAYS;
	}
	return VK_COMPARE_OP_ALWAYS;
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

	return hash;
}

static SK::VkRendererBackend::PipelineLayoutKey buildPipelineLayoutKey(
	SK::VkRendererBackend::VkRenderContext* vkRenderContext,
	bool usesSceneResources,
	bool usesMaterialResources,
	uint32_t pushConstantSize,
	SK::Renderer::ShaderStageFlags pushConstantStages)
{
	SK::VkRendererBackend::PipelineLayoutKey layoutKey{};

	if (usesSceneResources)
	{
		layoutKey.setLayouts.push_back(vkRenderContext->vkRendererBackend->gpuSceneDataDescriptorLayout);
	}

	if (usesMaterialResources)
	{
		layoutKey.setLayouts.push_back(vkRenderContext->sceneResources->vkMaterialRegistry.resourceDescriptorSetLayout);
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

static SK::Renderer::PipelineHandle getGraphicsPipeline_(SK::Renderer::RenderContext* renderContext, const SK::Renderer::GraphicsPipelineDesc& desc)
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
static SK::Renderer::PipelineHandle getComputePipeline_(SK::Renderer::RenderContext* renderContext, const SK::Renderer::ComputePipelineDesc& desc)
{
	return SK::Renderer::PipelineHandle{ SK::Renderer::INVALID_HANDLE };
}

static SK::Renderer::BufferDeviceAddress getVertexBufferDeviceAddress_(SK::Renderer::RenderContext* renderContext, size_t meshIndex)
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

static void beginMainRendering_(SK::Renderer::RenderContext* renderContext)
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

static void endRendering_(SK::Renderer::RenderContext* renderContext)
{
	SK::VkRendererBackend::VkRenderContext* vkRenderContext = fetchVkRenderContext(renderContext);
	SK::VkRendererBackend::State* vkRendererBackend = vkRenderContext->vkRendererBackend;

	vkCmdEndRendering(vkRendererBackend->currentCmdBuffer);
}

static void bindPipeline_(SK::Renderer::RenderContext* renderContext, SK::Renderer::PipelineHandle pipeline)
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

static void bindSceneResources_(SK::Renderer::RenderContext* renderContext)
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

static void bindMaterialResources_(SK::Renderer::RenderContext* renderContext)
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

static void pushConstants_(SK::Renderer::RenderContext* renderContext, SK::Renderer::ShaderStageFlags stages, uint32_t offset, uint32_t size, const void* data)
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

static void bindIndexBuffer_(SK::Renderer::RenderContext* renderContext, size_t meshIndex, SK::Renderer::IndexType indexType)
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

static void drawIndexed_(SK::Renderer::RenderContext* renderContext, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance)
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

static void dispatch_(SK::Renderer::RenderContext* renderContext, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
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

// --------------------------------Buffer------------------------------------------------------
static VkBufferUsageFlags toVkBufferUsageFlags(SK::Renderer::BufferUsage usage)
{
	VkBufferUsageFlags flags = 0;

	if (SK::Renderer::hasFlag(usage, SK::Renderer::BufferUsage::TransferSrc))          flags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	if (SK::Renderer::hasFlag(usage, SK::Renderer::BufferUsage::TransferDst))          flags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	if (SK::Renderer::hasFlag(usage, SK::Renderer::BufferUsage::UniformBuffer))        flags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	if (SK::Renderer::hasFlag(usage, SK::Renderer::BufferUsage::StorageBuffer))        flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	if (SK::Renderer::hasFlag(usage, SK::Renderer::BufferUsage::IndexBuffer))          flags |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
	if (SK::Renderer::hasFlag(usage, SK::Renderer::BufferUsage::VertexBuffer))         flags |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
	if (SK::Renderer::hasFlag(usage, SK::Renderer::BufferUsage::IndirectBuffer))       flags |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
	if (SK::Renderer::hasFlag(usage, SK::Renderer::BufferUsage::ShaderDeviceAddress))  flags |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

	if (SK::Renderer::hasFlag(usage, SK::Renderer::BufferUsage::AccelStructInput))
		flags |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

	if (SK::Renderer::hasFlag(usage, SK::Renderer::BufferUsage::AccelStructStorage))
		flags |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;

	if (SK::Renderer::hasFlag(usage, SK::Renderer::BufferUsage::ShaderBindingTable))
		flags |= VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR;

	return flags;
}

// Legacy-style mapping (VMA_MEMORY_USAGE_GPU_ONLY etc.). Still works, but VMA
// 3.x recommends VMA_MEMORY_USAGE_AUTO + explicit allocation flags instead
// TODO: Upgrade VMA to its recommended style.
static VmaMemoryUsage toVmaMemoryUsageLegacy(SK::Renderer::MemoryUsage usage)
{
	switch (usage)
	{
	case SK::Renderer::MemoryUsage::GpuOnly:  return VMA_MEMORY_USAGE_GPU_ONLY;
	case SK::Renderer::MemoryUsage::CpuOnly:  return VMA_MEMORY_USAGE_CPU_ONLY;
	case SK::Renderer::MemoryUsage::CpuToGpu: return VMA_MEMORY_USAGE_CPU_TO_GPU;
	case SK::Renderer::MemoryUsage::GpuToCpu: return VMA_MEMORY_USAGE_GPU_TO_CPU;
	case SK::Renderer::MemoryUsage::CpuCopy:  return VMA_MEMORY_USAGE_CPU_COPY;
	case SK::Renderer::MemoryUsage::Auto:    return VMA_MEMORY_USAGE_AUTO;
	}
	return VMA_MEMORY_USAGE_AUTO;
}

// --------------------------------Texture------------------------------------------------------
static VkExtent3D toVkExtent3D(SK::Renderer::Extent3D extent)
{
	return VkExtent3D{ extent.width, extent.height, extent.depth };
}

static VkFormat toVkFormat(SK::Renderer::Format format)
{
	switch (format)
	{
	case SK::Renderer::Format::Unknown:                    return VK_FORMAT_UNDEFINED;

	case SK::Renderer::Format::R8Unorm:                    return VK_FORMAT_R8_UNORM;
	case SK::Renderer::Format::R8Snorm:                    return VK_FORMAT_R8_SNORM;
	case SK::Renderer::Format::R8Uint:                     return VK_FORMAT_R8_UINT;
	case SK::Renderer::Format::R8Sint:                     return VK_FORMAT_R8_SINT;
	case SK::Renderer::Format::RG8Unorm:                   return VK_FORMAT_R8G8_UNORM;
	case SK::Renderer::Format::RG8Snorm:                   return VK_FORMAT_R8G8_SNORM;
	case SK::Renderer::Format::RG8Uint:                    return VK_FORMAT_R8G8_UINT;
	case SK::Renderer::Format::RG8Sint:                    return VK_FORMAT_R8G8_SINT;
	case SK::Renderer::Format::RGBA8Unorm:                 return VK_FORMAT_R8G8B8A8_UNORM;
	case SK::Renderer::Format::RGBA8UnormSrgb:             return VK_FORMAT_R8G8B8A8_SRGB;
	case SK::Renderer::Format::RGBA8Snorm:                 return VK_FORMAT_R8G8B8A8_SNORM;
	case SK::Renderer::Format::RGBA8Uint:                  return VK_FORMAT_R8G8B8A8_UINT;
	case SK::Renderer::Format::RGBA8Sint:                  return VK_FORMAT_R8G8B8A8_SINT;
	case SK::Renderer::Format::BGRA8Unorm:                 return VK_FORMAT_B8G8R8A8_UNORM;
	case SK::Renderer::Format::BGRA8UnormSrgb:             return VK_FORMAT_B8G8R8A8_SRGB;

	case SK::Renderer::Format::R16Unorm:                   return VK_FORMAT_R16_UNORM;
	case SK::Renderer::Format::R16Uint:                    return VK_FORMAT_R16_UINT;
	case SK::Renderer::Format::R16Sint:                    return VK_FORMAT_R16_SINT;
	case SK::Renderer::Format::R16Float:                   return VK_FORMAT_R16_SFLOAT;
	case SK::Renderer::Format::RG16Uint:                   return VK_FORMAT_R16G16_UINT;
	case SK::Renderer::Format::RG16Sint:                   return VK_FORMAT_R16G16_SINT;
	case SK::Renderer::Format::RG16Float:                  return VK_FORMAT_R16G16_SFLOAT;
	case SK::Renderer::Format::RGBA16Unorm:                return VK_FORMAT_R16G16B16A16_UNORM;
	case SK::Renderer::Format::RGBA16Uint:                 return VK_FORMAT_R16G16B16A16_UINT;
	case SK::Renderer::Format::RGBA16Sint:                 return VK_FORMAT_R16G16B16A16_SINT;
	case SK::Renderer::Format::RGBA16Float:                return VK_FORMAT_R16G16B16A16_SFLOAT;

	case SK::Renderer::Format::R32Uint:                    return VK_FORMAT_R32_UINT;
	case SK::Renderer::Format::R32Sint:                    return VK_FORMAT_R32_SINT;
	case SK::Renderer::Format::R32Float:                   return VK_FORMAT_R32_SFLOAT;
	case SK::Renderer::Format::RG32Uint:                   return VK_FORMAT_R32G32_UINT;
	case SK::Renderer::Format::RG32Sint:                   return VK_FORMAT_R32G32_SINT;
	case SK::Renderer::Format::RG32Float:                  return VK_FORMAT_R32G32_SFLOAT;
	case SK::Renderer::Format::RGB32Uint:                  return VK_FORMAT_R32G32B32_UINT;
	case SK::Renderer::Format::RGB32Sint:                  return VK_FORMAT_R32G32B32_SINT;
	case SK::Renderer::Format::RGB32Float:                 return VK_FORMAT_R32G32B32_SFLOAT;
	case SK::Renderer::Format::RGBA32Uint:                 return VK_FORMAT_R32G32B32A32_UINT;
	case SK::Renderer::Format::RGBA32Sint:                 return VK_FORMAT_R32G32B32A32_SINT;
	case SK::Renderer::Format::RGBA32Float:                return VK_FORMAT_R32G32B32A32_SFLOAT;

	// NOTE: same bit layout as DXGI_FORMAT_R10G10B10A2_UNORM, described
	// from the opposite byte order
	case SK::Renderer::Format::RGB10A2Unorm:               return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
	// Same story: matches DXGI_FORMAT_R11G11B10_FLOAT.
	case SK::Renderer::Format::RG11B10Float:               return VK_FORMAT_B10G11R11_UFLOAT_PACK32;

	case SK::Renderer::Format::Depth16Unorm:               return VK_FORMAT_D16_UNORM;
	// Not guaranteed supported on all Vulkan implementations - query
	// vkGetPhysicalDeviceFormatProperties before relying on this one.
	case SK::Renderer::Format::Depth24UnormStencil8Uint:   return VK_FORMAT_D24_UNORM_S8_UINT;
	case SK::Renderer::Format::Depth32Float:               return VK_FORMAT_D32_SFLOAT;
	case SK::Renderer::Format::Depth32FloatStencil8Uint:   return VK_FORMAT_D32_SFLOAT_S8_UINT;

	case SK::Renderer::Format::BC1RgbaUnorm:               return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
	case SK::Renderer::Format::BC1RgbaUnormSrgb:           return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
	case SK::Renderer::Format::BC3RgbaUnorm:               return VK_FORMAT_BC3_UNORM_BLOCK;
	case SK::Renderer::Format::BC3RgbaUnormSrgb:           return VK_FORMAT_BC3_SRGB_BLOCK;
	case SK::Renderer::Format::BC4RUnorm:                  return VK_FORMAT_BC4_UNORM_BLOCK;
	case SK::Renderer::Format::BC4RSnorm:                  return VK_FORMAT_BC4_SNORM_BLOCK;
	case SK::Renderer::Format::BC5RgUnorm:                 return VK_FORMAT_BC5_UNORM_BLOCK;
	case SK::Renderer::Format::BC5RgSnorm:                 return VK_FORMAT_BC5_SNORM_BLOCK;
	case SK::Renderer::Format::BC6HRgbUfloat:              return VK_FORMAT_BC6H_UFLOAT_BLOCK;
	case SK::Renderer::Format::BC6HRgbSfloat:              return VK_FORMAT_BC6H_SFLOAT_BLOCK;
	case SK::Renderer::Format::BC7RgbaUnorm:               return VK_FORMAT_BC7_UNORM_BLOCK;
	case SK::Renderer::Format::BC7RgbaUnormSrgb:           return VK_FORMAT_BC7_SRGB_BLOCK;
	}

	return VK_FORMAT_UNDEFINED;
}

static VkImageUsageFlags toVkImageUsageFlags(SK::Renderer::TextureUsage usage)
{
	VkImageUsageFlags flags = 0;

	if (SK::Renderer::hasFlag(usage, SK::Renderer::TextureUsage::TransferSrc))            flags |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	if (SK::Renderer::hasFlag(usage, SK::Renderer::TextureUsage::TransferDst))            flags |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	if (SK::Renderer::hasFlag(usage, SK::Renderer::TextureUsage::Sampled))                flags |= VK_IMAGE_USAGE_SAMPLED_BIT;
	if (SK::Renderer::hasFlag(usage, SK::Renderer::TextureUsage::Storage))                flags |= VK_IMAGE_USAGE_STORAGE_BIT;
	if (SK::Renderer::hasFlag(usage, SK::Renderer::TextureUsage::ColorAttachment))        flags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	if (SK::Renderer::hasFlag(usage, SK::Renderer::TextureUsage::DepthStencilAttachment)) flags |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

	return flags;
}

// --------------------------------Sampler------------------------------------------------------
static VkFilter toVkFilter(SK::Renderer::Filter f)
{
	return f == SK::Renderer::Filter::Linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
}

static VkSamplerMipmapMode toVkMipmapMode(SK::Renderer::MipmapMode m)
{
	return m == SK::Renderer::MipmapMode::Linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
}

static VkSamplerAddressMode toVkAddressMode(SK::Renderer::AddressMode mode)
{
	switch (mode)
	{
	case SK::Renderer::AddressMode::Repeat:            return VK_SAMPLER_ADDRESS_MODE_REPEAT;
	case SK::Renderer::AddressMode::MirroredRepeat:    return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
	case SK::Renderer::AddressMode::ClampToEdge:       return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	case SK::Renderer::AddressMode::ClampToBorder:     return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	case SK::Renderer::AddressMode::MirrorClampToEdge: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
	}
	return VK_SAMPLER_ADDRESS_MODE_REPEAT;
}

static VkBorderColor toVkBorderColor(SK::Renderer::BorderColor color)
{
	switch (color)
	{
	case SK::Renderer::BorderColor::TransparentBlack: return VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
	case SK::Renderer::BorderColor::OpaqueBlack:      return VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
	case SK::Renderer::BorderColor::OpaqueWhite:      return VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
	}
	return VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
}

static VkSamplerCreateInfo toVkSamplerCreateInfo(const SK::Renderer::SamplerDesc& desc)
{
	VkSamplerCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	info.magFilter = toVkFilter(desc.magFilter);
	info.minFilter = toVkFilter(desc.minFilter);
	info.mipmapMode = toVkMipmapMode(desc.mipmapMode);
	info.addressModeU = toVkAddressMode(desc.addressModeU);
	info.addressModeV = toVkAddressMode(desc.addressModeV);
	info.addressModeW = toVkAddressMode(desc.addressModeW);
	info.mipLodBias = desc.mipLodBias;
	info.anisotropyEnable = desc.anisotropyEnable ? VK_TRUE : VK_FALSE;
	info.maxAnisotropy = desc.maxAnisotropy;
	info.compareEnable = desc.compareEnable ? VK_TRUE : VK_FALSE;
	info.compareOp = toVkCompareOp(desc.compareOp);
	info.minLod = desc.minLod;
	info.maxLod = desc.maxLod;
	info.borderColor = toVkBorderColor(desc.borderColor);
	return info;
}

static SK::Renderer::BufferHandle createBuffer_(SK::Renderer::RenderContext* renderContext, const SK::Renderer::BufferDesc& desc)
{
	SK::VkRendererBackend::VkRenderContext* vkRenderContext = fetchVkRenderContext(renderContext);
	SK::VkRendererBackend::State* vkRendererBackend = vkRenderContext->vkRendererBackend;
	SK::VkRendererBackend::BufferRecord bufferRecord;
	bufferRecord.debugName = desc.debugName;

	if (!desc.data)
	{
		bufferRecord.buffer = SK::VkRendererBackend::createBuffer(vkRendererBackend, desc.size, toVkBufferUsageFlags(desc.usage), toVmaMemoryUsageLegacy(desc.memoryUsage));
	}
	else
	{
		bufferRecord.buffer = SK::VkRendererBackend::createAndUploadGPUBuffer(vkRendererBackend, desc.size, toVkBufferUsageFlags(desc.usage), desc.data);
	}

	const uint64_t bufferIndex = static_cast<uint64_t>(vkRenderContext->buffers.size());
	vkRenderContext->buffers.push_back(bufferRecord);
	
	return SK::Renderer::BufferHandle{ bufferIndex };
}

static size_t hashSamplerDesc(const SK::Renderer::SamplerDesc& desc)
{
	size_t hash = 0;

	std::hash<uint64_t> integerHasher;
	std::hash<float> floatHasher;

	hashCombine(&hash, integerHasher(static_cast<uint64_t>(desc.magFilter)));
	hashCombine(&hash, integerHasher(static_cast<uint64_t>(desc.minFilter)));
	hashCombine(&hash, integerHasher(static_cast<uint64_t>(desc.mipmapMode)));
	hashCombine(&hash, integerHasher(static_cast<uint64_t>(desc.addressModeU)));
	hashCombine(&hash, integerHasher(static_cast<uint64_t>(desc.addressModeV)));
	hashCombine(&hash, integerHasher(static_cast<uint64_t>(desc.addressModeW)));
	hashCombine(&hash, floatHasher(desc.mipLodBias));
	hashCombine(&hash, integerHasher(desc.anisotropyEnable ? 1 : 0));
	hashCombine(&hash, floatHasher(desc.maxAnisotropy));
	hashCombine(&hash, integerHasher(desc.compareEnable ? 1 : 0));
	hashCombine(&hash, floatHasher(desc.minLod));
	hashCombine(&hash, floatHasher(desc.maxLod));
	hashCombine(&hash, integerHasher(static_cast<uint64_t>(desc.borderColor)));

	return hash;
}

static uint32_t getOrCreateSampler(SK::Renderer::RenderContext* renderContext, const SK::Renderer::SamplerDesc& desc)
{
	SK::VkRendererBackend::VkRenderContext* vkRenderContext = fetchVkRenderContext(renderContext);
	SK::VkRendererBackend::State* vkRendererBackend = vkRenderContext->vkRendererBackend;

	const size_t descHash = hashSamplerDesc(desc);

	auto existing = vkRenderContext->samplerIndexByHash.find(descHash);
	if (existing != vkRenderContext->samplerIndexByHash.end())
	{
		return existing->second;
	}

	// Create and cache the sampler
	VkSamplerCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = toVkFilter(desc.magFilter),
		.minFilter = toVkFilter(desc.minFilter),
		.mipmapMode = toVkMipmapMode(desc.mipmapMode),
		.addressModeU = toVkAddressMode(desc.addressModeU),
		.addressModeV = toVkAddressMode(desc.addressModeV),
		.addressModeW = toVkAddressMode(desc.addressModeW),
		.mipLodBias = desc.mipLodBias,
		.anisotropyEnable = desc.anisotropyEnable,
		.maxAnisotropy = desc.maxAnisotropy,
		.compareEnable = desc.compareEnable,
		.compareOp = toVkCompareOp(desc.compareOp),
		.minLod = desc.minLod,
		.maxLod = desc.maxLod,
		.borderColor = toVkBorderColor(desc.borderColor)
	};

	VkSampler sampler = SK::VkRendererBackend::createSampler(vkRendererBackend, info);

	uint32_t samplerIndex = static_cast<uint32_t>(vkRenderContext->samplers.size());
	vkRenderContext->samplers.push_back(sampler);
	vkRenderContext->samplerIndexByHash[descHash] = samplerIndex;

	return samplerIndex;
}

static SK::Renderer::TextureHandle createTexture_(SK::Renderer::RenderContext* renderContext, const SK::Renderer::TextureDesc& textureDesc)
{
	SK::VkRendererBackend::VkRenderContext* vkRenderContext = fetchVkRenderContext(renderContext);
	SK::VkRendererBackend::State* vkRendererBackend = vkRenderContext->vkRendererBackend;

	SK::VkRendererBackend::TextureRecord textureRecord;
	textureRecord.debugName = textureDesc.debugName;
	if (textureDesc.data)
	{
		textureRecord.image = SK::VkRendererBackend::createImage(
			vkRendererBackend,
			textureDesc.data,
			textureDesc.dataSize,
			toVkExtent3D(textureDesc.imageExtent),
			toVkFormat(textureDesc.format),
			toVkImageUsageFlags(textureDesc.usage),
			textureDesc.mipMapped
		);
	}
	else
	{
		textureRecord.image = SK::VkRendererBackend::createImage(
			vkRendererBackend, 
			toVkExtent3D(textureDesc.imageExtent), 
			toVkFormat(textureDesc.format), 
			toVkImageUsageFlags(textureDesc.usage), 
			textureDesc.mipMapped);
	}

	if (textureDesc.samplerDesc.has_value())
	{
		textureRecord.samplerIndex = getOrCreateSampler(renderContext, textureDesc.samplerDesc.value());
	}

	const uint64_t textureIndex = static_cast<uint64_t>(vkRenderContext->textures.size());
	vkRenderContext->textures.push_back(textureRecord);

	return SK::Renderer::TextureHandle{ textureIndex };
}

void SK::VkRendererBackend::initVkRenderContext(VkRenderContext* vkRenderContext, State* vkRendererBackend, VkSceneResources* vkSceneResources)
{
	vkRenderContext->vkRendererBackend = vkRendererBackend;
	vkRenderContext->sceneResources = vkSceneResources;
	vkRenderContext->pipelines.clear();
	vkRenderContext->pipelineIndexByHash.clear();
	vkRenderContext->currentPipelineKind = SK::Renderer::PipelineKind::Graphics;
	vkRenderContext->currentPipelineLayout = VK_NULL_HANDLE;
	vkRenderContext->buffers.clear();
	vkRenderContext->textures.clear();
	vkRenderContext->samplers.clear();
	vkRenderContext->samplerIndexByHash.clear();
}

SK::Renderer::RenderContext SK::VkRendererBackend::makeRenderContext(VkRenderContext* vkRenderContext)
{
	static const SK::Renderer::RenderContextAPI api =
	{
		.getGraphicsPipeline = getGraphicsPipeline_,
		.getComputePipeline = getComputePipeline_,
		.beginMainRendering = beginMainRendering_,
		.endRendering = endRendering_,
		.bindPipeline = bindPipeline_,
		.bindSceneResources = bindSceneResources_,
		.bindMaterialResources = bindMaterialResources_,
		.pushConstants = pushConstants_,
		.bindIndexBuffer = bindIndexBuffer_,
		.drawIndexed = drawIndexed_,
		.dispatch = dispatch_,
		.getVertexBufferDeviceAddress = getVertexBufferDeviceAddress_,
		.createBuffer = createBuffer_,
		.createTexture = createTexture_
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
		for (BufferRecord& bufferRecord : vkRenderContext->buffers)
		{
			SK::VkRendererBackend::destroyBuffer(vkRenderContext->vkRendererBackend, bufferRecord.buffer);
		}

		for (TextureRecord& textureRecord : vkRenderContext->textures)
		{
			SK::VkRendererBackend::destroyImage(vkRenderContext->vkRendererBackend, textureRecord.image);
		}

		for (VkSampler sampler : vkRenderContext->samplers)
		{
			SK::VkRendererBackend::destroySampler(vkRenderContext->vkRendererBackend, sampler);
		}
	}

	vkRenderContext->vkRendererBackend = nullptr;
	vkRenderContext->sceneResources = nullptr;
	vkRenderContext->pipelines.clear();
	vkRenderContext->pipelineIndexByHash.clear();
	vkRenderContext->currentPipelineKind = SK::Renderer::PipelineKind::Graphics;
	vkRenderContext->currentPipelineLayout = VK_NULL_HANDLE;
	vkRenderContext->buffers.clear();
	vkRenderContext->textures.clear();
	vkRenderContext->samplers.clear();
	vkRenderContext->samplerIndexByHash.clear();
}

#pragma once

#include <cstdint>

namespace SK::Renderer
{
	static constexpr uint64_t INVALID_HANDLE = UINT64_MAX;

	// Opaque handle types
	struct PipelineHandle
	{
		uint64_t id;
	};

	struct BufferHandle
	{
		uint64_t id;
	};

	struct TextureHandle
	{
		uint64_t id;
	};

	// Using a named uint64_t for consistency with Graphics APIs. They use named aliases for uint64_t buffer device address such as VkDeviceAddress and D3D12_GPU_VIRTUAL_ADDRESS.
	using BufferDeviceAddress = uint64_t;

	using ShaderStageFlags = uint32_t;

	enum ShaderStageFlagBits : uint32_t
	{
		None = 0,
		VertexShader = 1 << 0,
		FragmentShader = 1 << 1,
		ComputeShader = 1 << 2,
	};

	enum class PipelineKind : uint8_t
	{
		Graphics = 0,
		Compute
	};

	enum class PrimitiveTopology : uint8_t
	{
		TriangleList = 0,
		LineList,
		PointList
	};

	enum class PolygonMode : uint8_t
	{
		Fill = 0,
		Line,
	};

	enum class CullMode : uint8_t
	{
		None = 0,
		Front,
		Back,
	};

	enum class FrontFace : uint8_t
	{
		Clockwise = 0,
		CounterClockwise,
	};

	enum class CompareOp : uint8_t
	{
		Never = 0,
		Less,
		LessEqual,
		Equal,
		GreaterEqual,
		Greater,
		Always,
	};

	enum class IndexType : uint8_t
	{
		Uint16 = 0,
		Uint32,
	};

	struct GraphicsPipelineDesc
	{
		const char* debugName = nullptr;

		const char* vertexShaderPath = nullptr;
		const char* fragmentShaderPath = nullptr;

		PrimitiveTopology topology = PrimitiveTopology::TriangleList;
		PolygonMode polygonMode = PolygonMode::Fill;
		CullMode cullMode = CullMode::None;
		FrontFace frontFace = FrontFace::CounterClockwise;

		bool depthTest = true;
		bool depthWrite = true;
		CompareOp depthCompare = CompareOp::LessEqual;

		bool blending = false;

		uint32_t pushConstantSize = 0;
		ShaderStageFlags pushConstantStages = ShaderStageFlagBits::None;

		bool usesSceneResources = false;
		bool usesMaterialResources = false;
	};

	struct ComputePipelineDesc
	{
		const char* debugName = nullptr;

		const char* computeShaderPath = nullptr;

		uint32_t pushConstantSize = 0;
		ShaderStageFlags pushConstantStages = ShaderStageFlagBits::None;

		bool usesSceneResources = false;
		bool usesMaterialResources = false;
	};

	struct RenderContext;

	struct RenderContextAPI
	{
		PipelineHandle (*getGraphicsPipeline)(RenderContext* renderContext, const GraphicsPipelineDesc& desc);
		// TODO: To be implemented
		PipelineHandle (*getComputePipeline)(RenderContext* renderContext, const ComputePipelineDesc& desc);

		// TODO: Also, implement a generic buffer device address getter.
		BufferDeviceAddress (*getVertexBufferDeviceAddress)(RenderContext* renderContext, size_t meshIndex);

		void (*beginMainRendering)(RenderContext* renderContext);
		void (*endRendering)(RenderContext* renderContext);

		void (*bindPipeline)(RenderContext* renderContext, PipelineHandle pipeline);

		void (*bindSceneResources)(RenderContext* renderContext);
		void (*bindMaterialResources)(RenderContext* renderContext);

		void (*pushConstants)(RenderContext* renderContext, ShaderStageFlags stages, uint32_t offset, uint32_t size, const void* data);

		void (*bindIndexBuffer)(RenderContext* renderContext, size_t meshIndex, IndexType indexType);
		void (*drawIndexed)(RenderContext* renderContext, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance);

		void (*dispatch)(RenderContext* renderContext, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ);
	};

	// Render Context packs up data (state) and functionality of the Graphics API backend. It provides functionality and hides the backend details.
	struct RenderContext
	{
		void* backend = nullptr; // backend state
		const RenderContextAPI* api = nullptr;
	};


	// Free function wrappers of function pointers in RenderContextAPI. This paradigm allows us to use free function style as well as providing a central entry point for checks and functions dispatches.
	PipelineHandle getGraphicsPipeline(RenderContext* renderContext, const GraphicsPipelineDesc& desc);
	PipelineHandle getComputePipeline(RenderContext* renderContext, const ComputePipelineDesc& desc);

	BufferDeviceAddress getVertexBufferDeviceAddress(RenderContext* renderContext, size_t meshIndex);

	void beginMainRendering(RenderContext* renderContext);
	void endRendering(RenderContext* renderContext);

	void bindPipeline(RenderContext* renderContext, PipelineHandle pipeline);

	void bindSceneResources(RenderContext* renderContext);
	void bindMaterialResources(RenderContext* renderContext);

	void pushConstants(RenderContext* renderContext, ShaderStageFlags stages, uint32_t offset, uint32_t size, const void* data);

	void bindIndexBuffer(RenderContext* renderContext, size_t meshIndex, IndexType indexType);
	void drawIndexed(RenderContext* renderContext, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance);

	void dispatch(RenderContext* renderContext, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ);
}
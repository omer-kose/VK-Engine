#pragma once

#include <cstdint>

namespace SK::Renderer
{
	static constexpr uint64_t INVALID_HANDLE = UINT64_MAX;

	// Opaque handle types
	// Handles are indices into records / resources in the Backend Render Context not actual values of backend resource handles.
	// They are opaque handle types, so caller side will be using them with the routines provided by the Render Context.
	struct PipelineHandle
	{
		uint64_t id;
	};

	struct ResourceSetHandle
	{
		uint64_t id;
	};

	struct PipelineResourceSet
	{
		ResourceSetHandle set;
		uint32_t slot;
	};

	struct BufferHandle
	{
		uint64_t id;
	};

	struct TextureHandle
	{
		uint64_t id;
	};

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
		Equal, 
		LessOrEqual, 
		Greater, 
		NotEqual, 
		GreaterOrEqual, 
		Always
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
		CompareOp depthCompare = CompareOp::LessOrEqual;

		bool blending = false;

		uint32_t pushConstantSize = 0;
		ShaderStageFlags pushConstantStages = ShaderStageFlagBits::None;

		// Engine-wide resources.
		// If true, scene resources are expected at set slot 0.
		bool usesSceneResources = false;
		// If true, material/bindless resources are expected at set slot 1.
		bool usesMaterialResources = false;

		// Renderer/pass-specific custom resources.
		// Each renderer decides the slot.
		std::vector<PipelineResourceSet> customResourceSets;
	};

	struct ComputePipelineDesc
	{
		const char* debugName = nullptr;

		const char* computeShaderPath = nullptr;

		uint32_t pushConstantSize = 0;
		ShaderStageFlags pushConstantStages = ShaderStageFlagBits::None;

		// Engine-wide resources.
		// If true, scene resources are expected at set slot 0.
		bool usesSceneResources = false;
		// If true, material/bindless resources are expected at set slot 1.
		bool usesMaterialResources = false;

		// Renderer/pass-specific custom resources.
		// Each renderer decides the slot.
		std::vector<PipelineResourceSet> customResourceSets;
	};

	// Generalized memory/allocation intent.
	// Describes how the resource will be accessed, not a literal memory type.
	// Each backend maps this to its own heap/pool concept.
	enum class MemoryUsage : uint8_t
	{
		GpuOnly,   // Device-local only, fastest GPU access, not CPU-visible
		CpuOnly,   // Host-visible, CPU reads/writes, e.g. readback targets
		CpuToGpu,  // Host-visible, optimized for frequent CPU writes read by GPU (staging/upload)
		GpuToCpu,  // Host-visible, optimized for GPU writes read back by CPU (readback)
		CpuCopy,   // Host-only staging memory, no device access at all
		Auto,      // Let the driver/allocator pick based on usage flags
	};

	// --------------------------------Buffer------------------------------------------------------
	// Generalized buffer usage bitmask.
	// Roughly maps to VkBufferUsageFlags on Vulkan. On D3D12 most of these bits
	// don't affect resource creation (they instead determine which view types
	// you're allowed to create later)
	enum class BufferUsage : uint32_t
	{
		None = 0,
		TransferSrc = 1u << 0,  // Source of a copy operation
		TransferDst = 1u << 1,  // Destination of a copy operation
		UniformBuffer = 1u << 2,  // Constant/uniform buffer (CBV in D3D12)
		StorageBuffer = 1u << 3,  // Read-write shader storage buffer (UAV in D3D12)
		IndexBuffer = 1u << 4,
		VertexBuffer = 1u << 5,
		IndirectBuffer = 1u << 6,  // Indirect draw/dispatch argument buffer
		ShaderDeviceAddress = 1u << 7,  // Buffer device address / raw GPU VA access
		AccelStructInput = 1u << 8,  // Input geometry for acceleration structure build
		AccelStructStorage = 1u << 9,  // Backing storage for an acceleration structure
		ShaderBindingTable = 1u << 10, // Ray tracing shader binding table
	};

	inline BufferUsage operator|(BufferUsage a, BufferUsage b)
	{
		return static_cast<BufferUsage>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
	}

	inline BufferUsage operator&(BufferUsage a, BufferUsage b)
	{
		return static_cast<BufferUsage>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
	}

	inline BufferUsage& operator|=(BufferUsage& a, BufferUsage b)
	{
		a = a | b;
		return a;
	}

	inline bool hasFlag(BufferUsage value, BufferUsage flag)
	{
		return (static_cast<uint32_t>(value) & static_cast<uint32_t>(flag)) == static_cast<uint32_t>(flag);
	}

	// Using a named uint64_t for consistency with Graphics APIs. They use named aliases for uint64_t buffer device address such as VkDeviceAddress and D3D12_GPU_VIRTUAL_ADDRESS.
	using BufferDeviceAddress = uint64_t;

	struct BufferDesc
	{
		size_t size; // size in bytes
		BufferUsage usage;
		MemoryUsage memoryUsage;
		const void* data = nullptr; // Initial data to be uploaded. When this is provided, the buffer will be chosen to be created on the GPU local memory.
		const char* debugName = nullptr;
	};

	// --------------------------------Texture------------------------------------------------------
	struct Extent3D
	{
		uint32_t width = 1;
		uint32_t height = 1;
		uint32_t depth = 1; // 3D textures only; leave at 1 for 2D/array/cube
	};

	enum class Format : uint16_t
	{
		Unknown,

		// 8-bit
		R8Unorm, R8Snorm, R8Uint, R8Sint,
		RG8Unorm, RG8Snorm, RG8Uint, RG8Sint,
		RGBA8Unorm, RGBA8UnormSrgb, RGBA8Snorm, RGBA8Uint, RGBA8Sint,
		BGRA8Unorm, BGRA8UnormSrgb, // common swapchain formats

		// 16-bit
		R16Unorm, R16Uint, R16Sint, R16Float,
		RG16Uint, RG16Sint, RG16Float,
		RGBA16Unorm, RGBA16Uint, RGBA16Sint, RGBA16Float,

		// 32-bit
		R32Uint, R32Sint, R32Float,
		RG32Uint, RG32Sint, RG32Float,
		RGB32Uint, RGB32Sint, RGB32Float, // vertex-attribute use mainly; not always renderable
		RGBA32Uint, RGBA32Sint, RGBA32Float,

		// Packed HDR
		RGB10A2Unorm,  // 10-10-10-2
		RG11B10Float,  // 11-11-10 float, common HDR render target

		// Depth / stencil
		Depth16Unorm,
		Depth24UnormStencil8Uint,
		Depth32Float,
		Depth32FloatStencil8Uint,

		// Block-compressed (desktop)
		BC1RgbaUnorm, BC1RgbaUnormSrgb,
		BC3RgbaUnorm, BC3RgbaUnormSrgb,
		BC4RUnorm, BC4RSnorm,
		BC5RgUnorm, BC5RgSnorm,
		BC6HRgbUfloat, BC6HRgbSfloat,
		BC7RgbaUnorm, BC7RgbaUnormSrgb,
	};

	enum class TextureUsage : uint32_t
	{
		None = 0,
		TransferSrc = 1u << 0,
		TransferDst = 1u << 1,
		Sampled = 1u << 2, // Read via a sampler in a shader (SRV in D3D12)
		Storage = 1u << 3, // Read-write access in a shader (UAV in D3D12)
		ColorAttachment = 1u << 4, // Render target
		DepthStencilAttachment = 1u << 5,
	};

	inline TextureUsage operator|(TextureUsage a, TextureUsage b)
	{
		return static_cast<TextureUsage>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
	}

	inline TextureUsage operator&(TextureUsage a, TextureUsage b)
	{
		return static_cast<TextureUsage>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
	}

	inline TextureUsage& operator|=(TextureUsage& a, TextureUsage b)
	{
		a = a | b;
		return a;
	}

	inline bool hasFlag(TextureUsage value, TextureUsage flag)
	{
		return (static_cast<uint32_t>(value) & static_cast<uint32_t>(flag)) == static_cast<uint32_t>(flag);
	}

	// --------------------------------Sampler------------------------------------------------------
	enum class Filter : uint8_t { Nearest, Linear };
	enum class MipmapMode : uint8_t { Nearest, Linear };

	enum class AddressMode : uint8_t
	{
		Repeat,            // VK_SAMPLER_ADDRESS_MODE_REPEAT             / D3D12_TEXTURE_ADDRESS_MODE_WRAP
		MirroredRepeat,    // VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT    / D3D12_TEXTURE_ADDRESS_MODE_MIRROR
		ClampToEdge,       // VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE      / D3D12_TEXTURE_ADDRESS_MODE_CLAMP
		ClampToBorder,     // VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER    / D3D12_TEXTURE_ADDRESS_MODE_BORDER
		MirrorClampToEdge, // VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE (Vulkan 1.2+) / D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE
	};

	enum class BorderColor : uint8_t
	{
		TransparentBlack,
		OpaqueBlack,
		OpaqueWhite,
	};

	struct SamplerDesc
	{
		Filter      magFilter = Filter::Linear;
		Filter      minFilter = Filter::Linear;
		MipmapMode  mipmapMode = MipmapMode::Linear;
		AddressMode addressModeU = AddressMode::Repeat;
		AddressMode addressModeV = AddressMode::Repeat;
		AddressMode addressModeW = AddressMode::Repeat;
		float       mipLodBias = 0.0f;
		bool        anisotropyEnable = false;
		float       maxAnisotropy = 1.0f;
		bool        compareEnable = false;
		CompareOp   compareOp = CompareOp::Always;
		float       minLod = 0.0f;
		float       maxLod = 1000.0f; // matches VK_LOD_CLAMP_NONE; treat as "no clamp" in practice
		BorderColor borderColor = BorderColor::TransparentBlack;
	};

	struct TextureDesc
	{
		Extent3D imageExtent;
		Format format;
		TextureUsage usage;
		std::optional<SamplerDesc> samplerDesc = std::nullopt; // if the texture is sampled, it will describe its sampler to be cached or retrieved.
		const char* debugName = nullptr;
		const void* data = nullptr; // initial data to be uploaded.
		size_t dataSize = 0; // will be filled when data is provided for the texture.
		bool mipMapped = false;
	};

	struct RenderContext;

	struct RenderContextAPI
	{
		PipelineHandle (*getGraphicsPipeline)(RenderContext* renderContext, const GraphicsPipelineDesc& desc);
		// TODO: To be implemented
		PipelineHandle (*getComputePipeline)(RenderContext* renderContext, const ComputePipelineDesc& desc);

		void (*beginMainRendering)(RenderContext* renderContext);
		void (*endRendering)(RenderContext* renderContext);

		void (*bindPipeline)(RenderContext* renderContext, PipelineHandle pipeline);

		void (*bindSceneResources)(RenderContext* renderContext);
		void (*bindMaterialResources)(RenderContext* renderContext);
		void (*bindResourceSet)(RenderContext* renderContext, uint32_t slot, ResourceSetHandle set);

		void (*pushConstants)(RenderContext* renderContext, ShaderStageFlags stages, uint32_t offset, uint32_t size, const void* data);

		void (*bindIndexBuffer)(RenderContext* renderContext, size_t meshIndex, IndexType indexType);
		void (*drawIndexed)(RenderContext* renderContext, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance);

		void (*dispatch)(RenderContext* renderContext, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ);

		// TODO: Also, implement a generic buffer device address getter.
		BufferDeviceAddress (*getVertexBufferDeviceAddress)(RenderContext* renderContext, size_t meshIndex);

		BufferHandle (*createBuffer)(RenderContext* renderContext, const BufferDesc& desc);
		TextureHandle(*createTexture)(RenderContext* renderContext, const TextureDesc& desc);
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

	void beginMainRendering(RenderContext* renderContext);
	void endRendering(RenderContext* renderContext);

	void bindPipeline(RenderContext* renderContext, PipelineHandle pipeline);

	void bindSceneResources(RenderContext* renderContext);
	void bindMaterialResources(RenderContext* renderContext);
	void bindResourceSet(RenderContext* renderContext, uint32_t slot, ResourceSetHandle set);

	void pushConstants(RenderContext* renderContext, ShaderStageFlags stages, uint32_t offset, uint32_t size, const void* data);

	void bindIndexBuffer(RenderContext* renderContext, size_t meshIndex, IndexType indexType);
	void drawIndexed(RenderContext* renderContext, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance);

	void dispatch(RenderContext* renderContext, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ);

	BufferDeviceAddress getVertexBufferDeviceAddress(RenderContext* renderContext, size_t meshIndex);

	BufferHandle createBuffer(RenderContext* renderContext, const BufferDesc& desc);
	TextureHandle createTexture(RenderContext* renderContext, const TextureDesc& desc);
}
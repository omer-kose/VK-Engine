/*
	Vulkan Renderer Backend
*/
#pragma once
#include <span>

#include <RendererBackend/Vulkan/VkTypes.h>
#include <RendererBackend/Vulkan/VkDescriptors.h>

#include <Util/DeletionQueue.h>

// Forward declarations
struct SDL_Window;
class Camera;

namespace SK::Renderer
{
	struct GPUSceneData;
	struct Vertex;
}

namespace SK::VkRendererBackend
{
	// Persistent resources that rotate and reused per frame
	struct FrameData
	{
		VkCommandPool commandPool;
		VkCommandBuffer mainCommandBuffer;

		VkSemaphore swapchainAcquireSemaphore;
		VkFence renderFence;

		DescriptorAllocatorGrowable frameDescriptorAllocator;

		// Per-Frame Resource Deletion Queue
		SK::Util::DeletionQueue deletionQueue;
	};

	struct RenderStats
	{
		float frameTime;
		int triangleCount;
		int drawCallCount;
		float sceneUpdateTime;
		float geometryDrawRecordTime;
	};

	constexpr unsigned int FRAME_OVERLAP = 2;

	struct PipelineLayoutKey
	{
		std::vector<VkDescriptorSetLayout> setLayouts;
		std::vector<VkPushConstantRange> pushConstantRanges;
	};

	struct PipelineKey
	{
		size_t vertShader;
		size_t fragShader;

		VkPrimitiveTopology topology;
		VkPolygonMode polygonMode;
		VkCullModeFlags cullMode;
		VkFrontFace frontFace;

		bool depthTest;
		bool depthWrite;
		VkCompareOp depthCompare;

		bool blending;

		VkFormat colorFormat;
		VkFormat depthFormat;

		VkPipelineLayout layout;
	};

	struct State
	{
		// Window related data stored (Window and other related params are owned by the App)
		SDL_Window* window{ nullptr }; // A non-owning ptr pointing to the window created by the App.
		VkExtent2D windowExtent{ }; // windowExtent is the window size determined by the application.

		bool isInitialized{ false };
		uint32_t frameNumber{ 0 };
		bool windowResizeRequested{ false };
		float renderScale{ 1.0f };
		// Vulkan Context
		VkInstance instance; // Vulkan library handle
		VkDebugUtilsMessengerEXT debugMessenger; // Vulkan debug output handle
		VkPhysicalDevice chosenGPU; // GPU chosen as the default device
		VkDevice device; // vulkan logical device for commands.
		VkSurfaceKHR surface; // Vulkan window surface

		// Swapchain 
		VkSwapchainKHR swapchain;
		VkFormat swapchainImageFormat;
		uint32_t numSwapchainImages;
		std::vector<VkImage> swapchainImages;
		std::vector<VkImageView> swapchainImageViews;
		// Always equal to the size of the window extent
		VkExtent2D swapchainExtent;

		// Synchronization Structures
		std::vector<VkSemaphore> submitSemaphores; 

		// Queues
		VkQueue graphicsQueue;
		uint32_t graphicsQueueFamily;

		// Allocator
		VmaAllocator vmaAllocator;

		// Frame Data
		FrameData frames[FRAME_OVERLAP];

		// Global Resource Deletion Queue
		SK::Util::DeletionQueue mainDeletionQueue;

		// Engine stats (TODO: Take this out)
		RenderStats stats;

		// Immeadiate submit structures
		VkFence immeadiateFence;
		VkCommandPool immediateCommandPool;
		VkCommandBuffer immediateCommandBuffer;

		// Main color attachment clear value 
		VkClearValue colorAttachmentClearValue = { 0.0f, 0.0f, 0.0f, 1.0f };

		// Draw Image
		AllocatedImage drawImage;
		AllocatedImage depthImage;
		VkExtent2D drawExtent;

		// Global Descriptors
		DescriptorAllocatorGrowable globalDescriptorAllocator;
		// Descriptor layout for single texture display
		VkDescriptorSetLayout displayTextureDescriptorSetLayout;
		// Scene Descriptor Layout (Global Descriptor Set 0 Layout)
		VkDescriptorSetLayout gpuSceneDataDescriptorLayout;

		// Per-frame Global Scene (uniform) Buffer and the descriptor set (Shared by the whole engine which uses scene data so it is persistent per-frame no need to reallocate) 
		AllocatedBuffer gpuSceneDataBuffer[FRAME_OVERLAP];
		VkDescriptorSet gpuSceneDescriptorSet[FRAME_OVERLAP];

		// Default textures
		AllocatedImage whiteImage;
		AllocatedImage blackImage;
		AllocatedImage greyImage;
		AllocatedImage errorCheckerboardImage;

		// Default samplers
		VkSampler defaultSamplerLinear;
		VkSampler defaultSamplerNearest;

		// Shader cache
		std::unordered_map<size_t, VkShaderModule> shaderCache;

		// Pipeline Layout cache
		std::unordered_map<size_t, VkPipelineLayout> pipelineLayoutCache;

		// Pipeline cache
		std::unordered_map<size_t, VkPipeline> pipelineCache;

		// Per-Frame Transient State (Filled by beginFrame function)
		uint32_t currentSwapchainImageIndex;
		VkCommandBuffer currentCmdBuffer;
		
	};


	// initializes everything in the renderer backend
	void init(State* vkRendererBackend, struct SDL_Window* window, uint32_t windowWidth, uint32_t windowHeight);

	// shuts down the renderer backend
	void shutdown(State* vkRendererBackend);

	// begin/end frames and some internal common draw functionalities
	bool beginFrame(State* vkRendererBackend);
	void endFrame(State* vkRendererBackend);

	void immediateSubmit(State* vkRendererBackend, std::function<void(VkCommandBuffer cmd)>&& function);

	AllocatedBuffer createBuffer(State* vkRendererBackend, size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
	// Allocates a buffer on local device memory and uploads the given data using a staging buffer
	AllocatedBuffer createAndUploadGPUBuffer(State* vkRendererBackend, size_t allocSize, VkBufferUsageFlags usage, const void* data, size_t srcOffset = 0, size_t dstOffset = 0);
	// Allocates a buffer on local device memory and uploads an already existing staging buffer on the CPU
	AllocatedBuffer uploadStagingBuffer(State* vkRendererBackend, VkBuffer stagingBuffer, size_t allocSize, VkBufferUsageFlags usage, size_t srcOffset = 0, size_t dstOffset = 0);
	void destroyBuffer(State* vkRendererBackend, const AllocatedBuffer& buffer);

	AllocatedImage createImage(State* vkRendererBackend, VkExtent3D imageExtent, VkFormat format, VkImageUsageFlags usage, bool mipMapped = false);
	AllocatedImage createImage(State* vkRendererBackend, void* data, VkExtent3D imageExtent, VkFormat format, VkImageUsageFlags usage, bool mipMapped = false);
	void destroyImage(State* vkRendererBackend, const AllocatedImage& img);

	VkSampler createSampler(State* vkRendererBackend, VkFilter minFilter, VkFilter magFilter, VkSamplerMipmapMode mipmapMode, VkSamplerAddressMode addressMode);

	VkGPUMeshBuffers uploadMesh(State* vkRendererBackend, std::span<SK::Renderer::Vertex> vertices, std::span<uint32_t> indices);

	void updateSceneBuffer(State* vkRendererBackend, const SK::Renderer::GPUSceneData& gpuSceneData);
	VkDescriptorSet fetchCurrentSceneBufferDescriptorSet(State* vkRendererBackend);

	void setViewport(State* vkRendererBackend, VkCommandBuffer cmd);
	void setScissor(State* vkRendererBackend, VkCommandBuffer cmd);

	VkShaderModule getOrLoadShader(State* vkRendererBackend, const char* path);
	void clearShaderCache(State* vkRendererBackend);

	VkPipelineLayout getOrCreatePipelineLayout(State* vkRendererBackend, const PipelineLayoutKey& key);
	void clearPipelineLayoutCache(State* vkRendererBackend);

	VkPipeline getOrCreatePipeline(State* vkRendererBackend, const PipelineKey& key);
	void clearPipelineCache(State* vkRendererBackend);

	FrameData& getCurrentFrameData(State* vkRendererBackend);

	void handleWindowResize(State* vkRendererBackend);

	/*
		Internal Helpers
	*/
	// Vulkan Context
	void initVulkan(State* vkRendererBackend);
	void initSwapchain(State* vkRendererBackend);
	void initCommands(State* vkRendererBackend);
	void initSyncStructures(State* vkRendererBackend);
	// Swapchain
	void createSwapchain(State* vkRendererBackend, uint32_t width, uint32_t height);
	void destroySwapchain(State* vkRendererBackend);
	// Descriptors
	void initDescriptors(State* vkRendererBackend);
	// Draw and Depth Images
	void createDrawAndDepthImages(State* vkRendererBackend);
	void destroyDrawAndDepthImages(State* vkRendererBackend);
	// Default Engine Data
	void initDefaultData(State* vkRendererBackend);
	// Init Scene Buffer
	void initGlobalSceneBuffer(State* vkRendererBackend);
};
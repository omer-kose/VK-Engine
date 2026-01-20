/*
	Vulkan Renderer Backend
*/
#pragma once

#include <RendererBackend/vulkan/vk_types.h>
#include <RendererBackend/vulkan/vk_descriptors.h>
#include <RendererBackend/vulkan/vk_loader.h>

#include <Pass/GLTFMetallicPass.h>

#include <Util/DeletionQueue.h>

// Forward declarations
struct SDL_Window;
class Camera;

namespace SK::VkRendererBackend
{
	// Persistent resources that rotate and reused per frame
	struct FrameData
	{
		VkCommandPool commandPool;
		VkCommandBuffer mainCommandBuffer;

		VkSemaphore swapchainSemaphore, renderSemaphore;
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

	/*
		Represents the geometry (and a possible material instance) of an object to be drawn in that frame. Created and destroyed per-frame.

		It can represent geometry from all kinds of formats.
	*/
	struct RenderObject
	{
		uint32_t indexCount;
		uint32_t firstIndex;
		VkBuffer indexBuffer;

		MaterialInstance* materialInstance; // a non-owning pointer

		Bounds bounds;

		glm::mat4 transform;
		VkDeviceAddress vertexBufferAddress;
	};

	/*
		Holds a flat list objects to be drawn that frame. The list is filled and reset every frame.

		For the time being, meshes coming from different formats are held in different lists so that the related passes can only fetch the required meshes and work with them.
	*/
	struct DrawContext
	{
		std::vector<RenderObject> opaqueGLTFSurfaces;
		std::vector<RenderObject> transparentGLTFSurfaces;

		void clear()
		{
			opaqueGLTFSurfaces.clear();
			transparentGLTFSurfaces.clear();
		}
	};

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

	/*
		Pass Context holds required information for a pass to use. It will be an opaque type for the passes.

		TODO: Name is too general 
	*/
	struct PassContext
	{
		VkCommandBuffer cmd;

		// Optional fields not all the passes needs them
		VkImageView targetImageView;
		VkExtent2D imageExtent;
		RendererBackend* vkRendererBackend; 
	};

	// TODO: Subject to change
	/*
		Reusable Passes that will be commonly used by all the programs like rendering UI, Gizmos etc. 
		Programs, will and can hold their own fields for UI for example but they don't have to manually render them. RendererBackend can render those automatically.
		So, programs using the vkRendererBackend framework can only focus on their own core pipelines and algorithms.
	*/
	
	// Such as UI, Gizmos etc.
	struct OverlayPass
	{
		void (*draw)(PassContext* passCtx);
	};


	struct RendererBackend
	{
		// Window related data stored (Window and other related params are owned by the App)
		SDL_Window* window{ nullptr }; // A non-owning ptr pointing to the window created by the App.
		VkExtent2D windowExtent{ }; // windowExtent is the window size determined by the application.

		bool isInitialized{ false };
		uint32_t frameNumber{ 0 };
		bool resizeRequested{ false };
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
		std::vector<VkImage> swapchainImages;
		std::vector<VkImageView> swapchainImageViews;
		VkExtent2D swapchainExtent;

		// Queues
		VkQueue graphicsQueue;
		uint32_t graphicsQueueFamily;

		// Allocator
		VmaAllocator vmaAllocator;

		// Frame Data and Queues
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
		// Main draw image descriptor used as the primary render target
		VkDescriptorSetLayout drawImageDescriptorSetLayout;
		VkDescriptorSet drawImageDescriptorSet;
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

		// Default materials
		MaterialInstance defaultMaterialInstance;

		std::vector<OverlayPass> overlayPasses;

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
	void init(RendererBackend* vkRendererBackend, struct SDL_Window* window, uint32_t windowWidth, uint32_t windowHeight);

	// shuts down the renderer backend
	void shutdown(RendererBackend* vkRendererBackend);

	// begin/end frames and some internal common draw functionalities
	bool beginFrame(RendererBackend* vkRendererBackend);
	void draw(RendererBackend* vkRendererBackend, const DrawContext& ctx, const GPUSceneData& gpuSceneData); // core draw loop
	void drawOverlays(RendererBackend* vkRendererBackend);
	void endFrame(RendererBackend* vkRendererBackend);

	void immediateSubmit(RendererBackend* vkRendererBackend, std::function<void(VkCommandBuffer cmd)>&& function);

	AllocatedBuffer createBuffer(RendererBackend* vkRendererBackend, size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
	void destroyBuffer(RendererBackend* vkRendererBackend, const AllocatedBuffer& buffer);

	AllocatedImage createImage(RendererBackend* vkRendererBackend, VkExtent3D imageExtent, VkFormat format, VkImageUsageFlags usage, bool mipMapped = false);
	AllocatedImage createImage(RendererBackend* vkRendererBackend, void* data, VkExtent3D imageExtent, VkFormat format, VkImageUsageFlags usage, bool mipMapped = false);
	void destroyImage(RendererBackend* vkRendererBackend, const AllocatedImage& img);

	GPUMeshBuffers uploadMesh(RendererBackend* vkRendererBackend, std::span<Vertex> vertices, std::span<uint32_t> indices);

	void updateSceneBuffer(RendererBackend* vkRendererBackend, const GPUSceneData& gpuSceneData);
	VkDescriptorSet fetchCurrentSceneBufferDescriptorSet(RendererBackend* vkRendererBackend);

	void setViewport(RendererBackend* vkRendererBackend, VkCommandBuffer cmd);
	void setScissor(RendererBackend* vkRendererBackend, VkCommandBuffer cmd);

	void registerOverlayPass(RendererBackend* vkRendererBackend, OverlayPass pass);

	VkShaderModule getOrLoadShader(RendererBackend* vkRendererBackend, const char* path);
	void clearShaderCache(RendererBackend* vkRendererBackend);

	VkPipelineLayout getOrCreatePipelineLayout(RendererBackend* vkRendererBackend, const PipelineLayoutKey& key);
	void clearPipelineLayoutCache(RendererBackend* vkRendererBackend);

	VkPipeline getOrCreatePipeline(RendererBackend* vkRendererBackend, const PipelineKey& key);
	void clearPipelineCache(RendererBackend* vkRendererBackend);

	FrameData& getCurrentFrameData(RendererBackend* vkRendererBackend);
	uint32_t getCurrentSwapchainImageIndex(RendererBackend* vkRendererBackend);
	VkCommandBuffer getCurrentCmdBuffer(RendererBackend* vkRendererBackend);

	/*
		Internal Helpers
		TODO: Rename these?
	*/
	// Vulkan Context
	void m_initVulkan(RendererBackend* vkRendererBackend);
	void m_initSwapchain(RendererBackend* vkRendererBackend);
	void m_initCommands(RendererBackend* vkRendererBackend);
	void m_initSyncStructures(RendererBackend* vkRendererBackend);
	// Swapchain
	void m_createSwapchain(RendererBackend* vkRendererBackend, uint32_t width, uint32_t height);
	void m_destroySwapchain(RendererBackend* vkRendererBackend);
	void m_resizeSwapchain(RendererBackend* vkRendererBackend);
	// Descriptors
	void m_initDescriptors(RendererBackend* vkRendererBackend);

	// Passes
	void m_initPasses(RendererBackend* vkRendererBackend);
	void m_clearPassResources(RendererBackend* vkRendererBackend);

	// Material Layouts
	void m_initMaterialLayouts(RendererBackend* vkRendererBackend);
	void m_clearMaterialLayouts(RendererBackend* vkRendererBackend);

	// Default Engine Data
	void m_initDefaultData(RendererBackend* vkRendererBackend);

	// Init Scene Buffer
	void m_initGlobalSceneBuffer(RendererBackend* vkRendererBackend);
};
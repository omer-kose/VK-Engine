/*
	Vulkan Renderer Backend
*/
#pragma once

#include <RendererBackend/vk_types.h>
#include <RendererBackend/vk_descriptors.h>
#include <RendererBackend/vk_loader.h>

#include <Pass/GLTFMetallicPass.h>

#include <Util/DeletionQueue.h>

// Forward declarations
struct SDL_Window;
class Camera;

namespace SK::VkRendererBackend
{
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
		Renderer* renderer; 
	};

	// TODO: Subject to change
	/*
		Reusable Passes that will be commonly used by all the programs like rendering UI, Gizmos etc. 
		Programs, will and can hold their own fields for UI for example but they don't have to manually render them. Renderer can render those automatically.
		So, programs using the renderer framework can only focus on their own core pipelines and algorithms.
	*/
	
	// Such as UI, Gizmos etc.
	struct OverlayPass
	{
		void (*draw)(PassContext* passCtx);
	};


	struct Renderer
	{
		// Window related data stored (Window and other related params are owned by the App)
		SDL_Window* window{ nullptr }; // A non-owning ptr pointing to the window created by the App.
		VkExtent2D windowExtent{ }; // windowExtent is the window size determined by the application.

		bool isInitialized{ false };
		uint32_t frameNumber{ 0 };
		bool freezeRendering{ false };
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

		// Engine stats
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
	};


	// initializes everything in the engine
	void init(Renderer* renderer, struct SDL_Window* window, uint32_t windowWidth, uint32_t windowHeight);

	// shuts down the renderer
	void shutdown(Renderer* renderer);

	// draw functionality
	void draw(Renderer* renderer, const DrawContext& ctx, const GPUSceneData& gpuSceneData); // core draw loop
	void drawMain(Renderer* renderer, VkCommandBuffer cmd, const DrawContext& ctx, const GPUSceneData& gpuSceneData); // function to simplify the main draw function. It handles some transitions, attachments and calls to actualy drawing functionality below
	void drawGeometry(Renderer* renderer, VkCommandBuffer cmd, const DrawContext& ctx);

	void immediateSubmit(Renderer* renderer, std::function<void(VkCommandBuffer cmd)>&& function);

	// Renderer Utilities TODO: I think renderer shouldn't be responsible of providing such functionalities. The only reason of these being here is that renderer holds vmaAllocator  
	AllocatedBuffer createBuffer(Renderer* renderer, size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
	void destroyBuffer(Renderer* renderer, const AllocatedBuffer& buffer);

	AllocatedImage createImage(Renderer* renderer, VkExtent3D imageExtent, VkFormat format, VkImageUsageFlags usage, bool mipMapped = false);
	AllocatedImage createImage(Renderer* renderer, void* data, VkExtent3D imageExtent, VkFormat format, VkImageUsageFlags usage, bool mipMapped = false);
	void destroyImage(Renderer* renderer, const AllocatedImage& img);

	GPUMeshBuffers uploadMesh(Renderer* renderer, std::span<Vertex> vertices, std::span<uint32_t> indices);

	void updateSceneBuffer(Renderer* renderer, const GPUSceneData& gpuSceneData);
	VkDescriptorSet fetchCurrentSceneBufferDescriptorSet(Renderer* renderer);

	void setViewport(Renderer* renderer, VkCommandBuffer cmd);
	void setScissor(Renderer* renderer, VkCommandBuffer cmd);

	FrameData& fetchCurrentFrameData(Renderer* renderer);

	void registerOverlayPass(Renderer* renderer, OverlayPass pass);

	/*
		Internal Helpers
	*/
	// Vulkan Context
	void m_initVulkan(Renderer* renderer);
	void m_initSwapchain(Renderer* renderer);
	void m_initCommands(Renderer* renderer);
	void m_initSyncStructures(Renderer* renderer);
	// Swapchain
	void m_createSwapchain(Renderer* renderer, uint32_t width, uint32_t height);
	void m_destroySwapchain(Renderer* renderer);
	void m_resizeSwapchain(Renderer* renderer);
	// Descriptors
	void m_initDescriptors(Renderer* renderer);

	// Passes
	void m_initPasses(Renderer* renderer);
	void m_clearPassResources(Renderer* renderer);

	// Material Layouts
	void m_initMaterialLayouts(Renderer* renderer);
	void m_clearMaterialLayouts(Renderer* renderer);

	// Default Engine Data
	void m_initDefaultData(Renderer* renderer);

	// Init Scene Buffer
	void m_initGlobalSceneBuffer(Renderer* renderer);
};
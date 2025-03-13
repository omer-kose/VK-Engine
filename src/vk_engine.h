#pragma once

#include <vk_types.h>
#include <vk_descriptors.h>
#include <vk_loader.h>


struct DeletionQueue
{
	void pushFunction(std::function<void()>&& function)
	{
		deletors.push_back(function);
	}

	void flush()
	{
		// Reverse iterate the deletion queue to execute all the deletion functions
		for(auto it = deletors.rbegin(); it != deletors.rend(); ++it)
		{
			(*it)();
		}

		deletors.clear();
	}

	std::deque<std::function<void()>> deletors;

};

struct FrameData
{
	VkCommandPool commandPool;
	VkCommandBuffer mainCommandBuffer;

	VkSemaphore swapchainSemaphore, renderSemaphore;
	VkFence renderFence;
	
	DescriptorAllocatorGrowable frameDescriptorAllocator;

	// Per-Frame Resource Deletion Queue
	DeletionQueue deletionQueue;
};

constexpr unsigned int FRAME_OVERLAP = 2;

struct ComputePushConstants
{
	glm::vec4 data1;
	glm::vec4 data2;
	glm::vec4 data3;
	glm::vec4 data4;
};

struct ComputeEffect
{
	const char* name;

	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;

	ComputePushConstants pushConstants;
};

class VulkanEngine {
public:
	static VulkanEngine& Get();

	//initializes everything in the engine
	void init();

	//shuts down the engine
	void cleanup();

	//draw loop
	void draw();
	void drawBackground(VkCommandBuffer cmd);
	void drawGeometry(VkCommandBuffer cmd);
	void drawImgui(VkCommandBuffer cmd, VkImageView targetImageView);

	//run main loop
	void run();

	void immediateSubmit(std::function<void(VkCommandBuffer cmd)>&& function);

	AllocatedBuffer createBuffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
	void destroyBuffer(const AllocatedBuffer& buffer);

	AllocatedImage createImage(VkExtent3D imageExtent, VkFormat format, VkImageUsageFlags usage, bool mipMapped = false);
	AllocatedImage createImage(void* data, VkExtent3D imageExtent, VkFormat format, VkImageUsageFlags usage, bool mipMapped = false);
	void destroyImage(const AllocatedImage& img);

	GPUMeshBuffers uploadMesh(std::span<Vertex> vertices, std::span<uint32_t> indices);

public:
	struct SDL_Window* window{ nullptr };

	bool isInitialized{ false };
	int frameNumber{0};
	bool freezeRendering{ false };
	bool resizeRequested{ false };
	float renderScale{ 1.0f };
	VkExtent2D windowExtent{ 1920 , 1080 };
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

	// Frame Data and Queues
	FrameData frames[FRAME_OVERLAP];
	FrameData& getCurrentFrame() { return frames[frameNumber % FRAME_OVERLAP]; }

	VkQueue graphicsQueue;
	uint32_t graphicsQueueFamily;

	// Global Resource Deletion Queue
	DeletionQueue mainDeletionQueue;

	// Allocator
	VmaAllocator vmaAllocator;

	// Draw Image
	AllocatedImage drawImage;
	AllocatedImage depthImage;
	VkExtent2D drawExtent;

	// Global Descriptors
	DescriptorAllocator globalDescriptorAllocator;
	VkDescriptorSetLayout drawImageDescriptorSetLayout;
	VkDescriptorSet drawImageDescriptorSet;

	// Background Compute Pipelines
	VkPipelineLayout gradientPipelineLayout; // all of the effects share the same layout so we only create one
	std::vector<ComputeEffect> backgroundEffects;
	int currentBackgroundEffect{0};

	/* Graphics Pipelines */
	// Mesh Pipeline
	VkPipelineLayout meshPipelineLayout;
	VkPipeline meshPipeline;

	// Test Meshes
	std::vector<std::shared_ptr<MeshAsset>> testMeshes;

	// Draw Resources
	GPUSceneData sceneData;
	// Draw Resource Descriptor Layouts
	VkDescriptorSetLayout sceneDataDescriptorLayout;

	// Immeadiate submit structures
	VkFence immeadiateFence;
	VkCommandPool immediateCommandPool;
	VkCommandBuffer immediateCommandBuffer;
private:
	// Vulkan Context
	void m_initVulkan();
	void m_initSwapchain();
	void m_initCommands();
	void m_initSyncStructures();
	// Swapchain
	void m_createSwapchain(uint32_t width, uint32_t height);
	void m_destroySwapchain();
	void m_resizeSwapchain();
	// Descriptors
	void m_initDescriptors();
	// Pipelines
	void m_initPipelines();
	void m_initBackgroundPipelines();
	void m_initMeshPipeline();

	// ImGui
	void m_initImgui();

	// Test Data
	void m_initDefaultData();
};

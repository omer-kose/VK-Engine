#pragma once

#include <vk_types.h>
#include <vk_descriptors.h>
#include <vk_loader.h>

#include <camera.h>

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

// PBR Metallic Material follows the GLTF format
struct GLTFMetallicRoughnessMaterial
{
	MaterialPipeline opaquePipeline;
	MaterialPipeline transparentPipeline;

	VkDescriptorSetLayout materialLayout;

	// CPU representation of the MaterialConstants uniform buffer
	struct MaterialConstants
	{
		glm::vec4 colorFactors;
		glm::vec4 metalRoughnessFactors;
		// padding to complete the uniform buffer to 256 bytes (most GPUs expect a minimum alignment of 256 bytes for uniform buffers)
		glm::vec4 extra[14];
	};

	struct MaterialResources
	{
		AllocatedImage colorImage;
		VkSampler colorSampler;
		AllocatedImage metalRoughnessImage;
		VkSampler metalRoughnessSampler;
		VkBuffer dataBuffer; // Handle to the buffer holding MaterialConstants data
		uint32_t dataBufferOffset; // Multiple materials in a GLTF files will be stored in a single buffer, so the actual data for the specific material instance is fetched with this offset
	};

	DescriptorWriter writer;

	void buildPipelines(VulkanEngine* engine);
	// This struct only stores the pipelines and the layouts. The material resources are allocated outside. Allocator must clean them properly.
	void clearResources(VkDevice device);

	MaterialInstance createInstance(VkDevice device, MaterialPass pass, const MaterialResources& resources, DescriptorAllocatorGrowable& descriptorAllocator);
};

struct RenderObject
{
	uint32_t indexCount;
	uint32_t firstIndex;
	VkBuffer indexBuffer;

	MaterialInstance* material;

	glm::mat4 transform;
	VkDeviceAddress vertexBufferAddress;
};

struct DrawContext
{
	std::vector<RenderObject> opaqueSurfaces;
};

struct MeshNode : public SceneNode
{
	std::shared_ptr<MeshAsset> mesh;

	// Creates and adds all the surfaces in the mesh into the context's opaqueSurfaces
	virtual void registerDraw(const glm::mat4& topMatrix, DrawContext& ctx) override;
};

class VulkanEngine {
public:
	static VulkanEngine& Get();

	// initializes everything in the engine
	void init();

	// shuts down the engine
	void cleanup();

	// draw functionality
	void draw();
	void drawBackground(VkCommandBuffer cmd);
	void drawGeometry(VkCommandBuffer cmd);
	void drawImgui(VkCommandBuffer cmd, VkImageView targetImageView);

	// Updates
	void updateScene();

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

	// Queues
	VkQueue graphicsQueue;
	uint32_t graphicsQueueFamily;

	// Allocator
	VmaAllocator vmaAllocator;

	// Frame Data and Queues
	FrameData frames[FRAME_OVERLAP];
	FrameData& getCurrentFrame() { return frames[frameNumber % FRAME_OVERLAP]; }

	// Global Resource Deletion Queue
	DeletionQueue mainDeletionQueue;

	// Immeadiate submit structures
	VkFence immeadiateFence;
	VkCommandPool immediateCommandPool;
	VkCommandBuffer immediateCommandBuffer;

	// Camera
	Camera mainCamera;

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

	// Background Compute Pipelines
	VkPipelineLayout gradientPipelineLayout; // all of the effects share the same layout so we only create one
	std::vector<ComputeEffect> backgroundEffects;
	int currentBackgroundEffect{0};

	/* Graphics Pipelines */
	// Mesh Pipeline
	VkPipelineLayout meshPipelineLayout;
	VkPipeline meshPipeline;

	// Default textures
	AllocatedImage whiteImage;
	AllocatedImage blackImage;
	AllocatedImage greyImage;
	AllocatedImage errorCheckerboardImage;
	
	// Default samplers
	VkSampler defaultSamplerLinear;
	VkSampler defaultSamplerNearest;	

	// Default materials
	GLTFMetallicRoughnessMaterial metallicRoughnessMaterial;
	MaterialInstance defaultMaterialInstance;

	// Main Draw Context
	DrawContext mainDrawContext;

	// Loaded scenes
	std::unordered_map<std::string, std::shared_ptr<LoadedGLTF>> loadedScenes;

	// Draw Resources
	GPUSceneData sceneData;
	// Draw Resource Descriptor Layouts
	VkDescriptorSetLayout sceneDataDescriptorLayout;
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

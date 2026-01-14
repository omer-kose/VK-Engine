#include "vk_renderer.h"

#include <chrono>
#include <thread>

#include <SDL_vulkan.h>

#include <RendererBackend/vulkan/vk_initializers.h>
#include <RendererBackend/vulkan/vk_types.h>
#include <RendererBackend/vulkan/vk_images.h>
#include <RendererBackend/vulkan/vk_pipelines.h>
#include "VkBootstrap.h"

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include "camera.h"

#include <glm/gtx/transform.hpp>

constexpr bool useValidationLayers = true;

static size_t hashPipelineKey(const SK::VkRendererBackend::PipelineKey& k)
{
    size_t h = 0;
    auto hc = [&](auto v)
    {
        std::hash<std::decay_t<decltype(v)>> hasher;
        h ^= hasher(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
    };

    hc(k.vertShader);
    hc(k.fragShader);
    hc(k.topology);
    hc(k.polygonMode);
    hc(k.cullMode);
    hc(k.frontFace);
    hc(k.depthTest);
    hc(k.depthWrite);
    hc(k.depthCompare);
    hc(k.blending);
    hc(k.colorFormat);
    hc(k.depthFormat);
    hc((uint64_t)k.layout);

    return h;
}


void SK::VkRendererBackend::init(Renderer* renderer, struct SDL_Window* window, uint32_t windowWidth, uint32_t windowHeight)
{
    // only one renderer initialization is allowed with the application.
    assert(renderer->isInitialized == false);

    // Store window data coming from the App for later use
    renderer->window = window;
    renderer->windowExtent = VkExtent2D{ windowWidth, windowHeight };

    // Vulkan Bootstrapping
    m_initVulkan(renderer);
    m_initSwapchain(renderer);
    m_initCommands(renderer);
    m_initSyncStructures(renderer);

    m_initDescriptors(renderer);

    m_initMaterialLayouts(renderer);

    m_initPasses(renderer);

    m_initDefaultData(renderer);

    m_initGlobalSceneBuffer(renderer);

    // everything went fine
    renderer->isInitialized = true;
}

void SK::VkRendererBackend::shutdown(Renderer* renderer)
{
    if(renderer->isInitialized) 
    {
        for(int i = 0; i < FRAME_OVERLAP; ++i)
        {
            // Destroy sync objects
            vkDestroyFence(renderer->device, renderer->frames[i].renderFence, nullptr);
            vkDestroySemaphore(renderer->device, renderer->frames[i].swapchainSemaphore, nullptr);
            vkDestroySemaphore(renderer->device, renderer->frames[i].renderSemaphore, nullptr);

            // It’s not possible to individually destroy VkCommandBuffer, destroying their parent pool will destroy all of the command buffers allocated from it.
            vkDestroyCommandPool(renderer->device, renderer->frames[i].commandPool, nullptr);

            renderer->frames[i].deletionQueue.flush();
        }

        // Clear out the caches
        clearShaderCache(renderer);
        clearPipelineCache(renderer);

        m_clearMaterialLayouts(renderer);

        m_clearPassResources(renderer);

        renderer->mainDeletionQueue.flush();

        m_destroySwapchain(renderer);

        vkDestroyDevice(renderer->device, nullptr);
        vkDestroySurfaceKHR(renderer->instance, renderer->surface, nullptr);

        vkb::destroy_debug_utils_messenger(renderer->instance, renderer->debugMessenger);
        vkDestroyInstance(renderer->instance, nullptr);

        // Nullify non-owning pointers
        renderer->window = nullptr;
        
        renderer->isInitialized = false;
    }
}

void SK::VkRendererBackend::draw(Renderer* renderer, const DrawContext& ctx, const GPUSceneData& sceneData)
{
    FrameData& currentFrame = fetchCurrentFrameData(renderer);
    // Wait until the GPU has finished rendering the last frame of the same modularity (0->1->2->3  wait on 2 for 0 and wait on 3 for 1 and so on)
    VK_CHECK(vkWaitForFences(renderer->device, 1, &currentFrame.renderFence, true, 1000000000));

    currentFrame.deletionQueue.flush();
    currentFrame.frameDescriptorAllocator.clearPools(renderer->device);

    // To be able to use the same fence it must be reset after use
    VK_CHECK(vkResetFences(renderer->device, 1, &currentFrame.renderFence));
    
    // Request an available image from the swapchain. swapchainSemaphore is signaled once it has finished presenting the image so it can be used again.
    // More detailed description of how vkAcquireNextImageKHR works: https://stackoverflow.com/questions/60419749/why-does-vkacquirenextimagekhr-never-block-my-thread
    uint32_t swapchainImageIndex;
    VkResult acquireResult = vkAcquireNextImageKHR(renderer->device, renderer->swapchain, 1000000000, currentFrame.swapchainSemaphore, nullptr, &swapchainImageIndex);
    if(acquireResult == VK_ERROR_OUT_OF_DATE_KHR)
    {
        renderer->resizeRequested = true;
        return;
    }

    // Extent of the image that we are going to draw onto
    renderer->drawExtent.width = std::min(renderer->drawImage.imageExtent.width, renderer->swapchainExtent.width) * renderer->renderScale;
    renderer->drawExtent.height = std::min(renderer->drawImage.imageExtent.height, renderer->swapchainExtent.height) * renderer->renderScale;
    
    // Vulkan handles are just a 64 bit handles/pointers, so its fine to copy them around, but remember that their actual data is handled by vulkan itself.
    VkCommandBuffer cmd = currentFrame.mainCommandBuffer;

    // Now we are sure that command is executed, we can safely reset it and begin recording again
    VK_CHECK(vkResetCommandBuffer(cmd, 0));

    // Begin the command buffer recording. We will submit this command buffer exactly once, so we let Vulkan know that
    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    // Start the command buffer recording
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    // Transition depth image to optimal depth layout
    vkutil::transitionImage(cmd, renderer->depthImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    
    // encode main drawing commands 
    drawMain(renderer, cmd, ctx, sceneData);

    // Transition the draw image and the swapchain image into their correct layouts
    vkutil::transitionImage(cmd, renderer->drawImage.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkutil::transitionImage(cmd, renderer->swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // Execute a copy operation from the draw image into the swapchain image
    vkutil::copyImageToImage(cmd, renderer->drawImage.image, renderer->swapchainImages[swapchainImageIndex], renderer->drawExtent, renderer->swapchainExtent);

    // After drawing, we need to draw overlays on top of the swapchain image, so transition the swapchain image into optimal drawing layout
    vkutil::transitionImage(cmd, renderer->swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // Execute overlay passes
    for(auto& pass : renderer->overlayPasses)
    {
        PassContext ctx = { cmd, renderer->swapchainImageViews[swapchainImageIndex], renderer->swapchainExtent };
        pass.draw(&ctx);
    }

    // Transition swapchain image into the presentation layout
    vkutil::transitionImage(cmd, renderer->swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    // Finalize the command buffer
    VK_CHECK(vkEndCommandBuffer(cmd));

    // Prepare the submission
    // We will wait on the swapchainSemaphore before executing the commands as that semaphore is signaled once swapchain is done presenting that image
    // We will signal renderSemaphore to signal that rendering has finished
    VkCommandBufferSubmitInfo cmdSubmitInfo = vkinit::command_buffer_submit_info(cmd);
    VkSemaphoreSubmitInfo waitInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, currentFrame.swapchainSemaphore);
    VkSemaphoreSubmitInfo signalInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, currentFrame.renderSemaphore);
    VkSubmitInfo2 submit = vkinit::submit_info(&cmdSubmitInfo, &signalInfo, &waitInfo);

    // Submit command buffer to the queue and execute it
    // renderFence will be signaled once the submitted command buffer has completed execution.
    VK_CHECK(vkQueueSubmit2(renderer->graphicsQueue, 1, &submit, currentFrame.renderFence));

    // Prepare the presentation
    // We will wait on the renderSemaphore so that it will be guaranteed that the rendering has been finished and the swapchain image is ready to be presented
    VkPresentInfoKHR presentInfo = {.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, .pNext = nullptr};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &renderer->swapchain;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &currentFrame.renderSemaphore;
    presentInfo.pImageIndices = &swapchainImageIndex;

    VkResult presentResult = vkQueuePresentKHR(renderer->graphicsQueue, &presentInfo);
    if(presentResult == VK_ERROR_OUT_OF_DATE_KHR)
    {
        renderer->resizeRequested = true;
    }

    // Increase the number of frames drawn
    ++renderer->frameNumber;
}

void SK::VkRendererBackend::drawMain(Renderer* renderer, VkCommandBuffer cmd, const DrawContext& ctx, const GPUSceneData& sceneData)
{
    updateSceneBuffer(renderer, sceneData);

    // When rendering geometry we need to use COLOR_ATTACHMENT_OPTIMAL as it is the most optimal layout for rendering with graphics pipeline
    vkutil::transitionImage(cmd, renderer->drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // Begin a renderpass connected to the draw image
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(renderer->drawImage.imageView, &renderer->colorAttachmentClearValue, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo depthAttachment = vkinit::depth_attachment_info(renderer->depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    VkRenderingInfo renderInfo = vkinit::rendering_info(renderer->drawExtent, &colorAttachment, &depthAttachment);
    vkCmdBeginRendering(cmd, &renderInfo);

    auto start = std::chrono::system_clock::now();

    drawGeometry(renderer, cmd, ctx);

    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    renderer->stats.geometryDrawRecordTime = elapsed.count() / 1000.f;

    vkCmdEndRendering(cmd);
}

void SK::VkRendererBackend::drawGeometry(Renderer* renderer, VkCommandBuffer cmd, const DrawContext& ctx)
{
    // Go through all the graphics passes and execute them
    GLTFMetallicPass::Execute(renderer, cmd, ctx);
}

void SK::VkRendererBackend::immediateSubmit(Renderer* renderer, std::function<void(VkCommandBuffer cmd)>&& function)
{
    // Before starting submitting and waiting on the fence reset them
    VK_CHECK(vkResetFences(renderer->device, 1, &renderer->immeadiateFence));
    VK_CHECK(vkResetCommandBuffer(renderer->immediateCommandBuffer, 0));
    // Prepare the immediate command buffer for executing function given as the param
    VkCommandBuffer cmd = renderer->immediateCommandBuffer;
    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));
    function(cmd);
    VK_CHECK(vkEndCommandBuffer(cmd));

    // Submit
    VkCommandBufferSubmitInfo cmdSubmitInfo = vkinit::command_buffer_submit_info(cmd);
    VkSubmitInfo2 submitInfo = vkinit::submit_info(&cmdSubmitInfo, nullptr, nullptr);
    VK_CHECK(vkQueueSubmit2(renderer->graphicsQueue, 1, &submitInfo, renderer->immeadiateFence));

    // Wait on the fence until the command buffer finished executing
    VK_CHECK(vkWaitForFences(renderer->device, 1, &renderer->immeadiateFence, true, 9999999999));
}

AllocatedBuffer SK::VkRendererBackend::createBuffer(Renderer* renderer, size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage)
{
    // Allocate buffer
    VkBufferCreateInfo bufferInfo = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = nullptr };
    bufferInfo.size = allocSize;
    bufferInfo.usage = usage;

    VmaAllocationCreateInfo vmaAllocInfo = {};
    vmaAllocInfo.usage = memoryUsage;
    vmaAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    AllocatedBuffer newBuffer;

    VK_CHECK(vmaCreateBuffer(renderer->vmaAllocator, &bufferInfo, &vmaAllocInfo, &newBuffer.buffer, &newBuffer.allocation, &newBuffer.allocInfo));

    return newBuffer;
}

void SK::VkRendererBackend::destroyBuffer(Renderer* renderer, const AllocatedBuffer& buffer)
{
    vmaDestroyBuffer(renderer->vmaAllocator, buffer.buffer, buffer.allocation);
}

AllocatedImage SK::VkRendererBackend::createImage(Renderer* renderer, VkExtent3D imageExtent, VkFormat format, VkImageUsageFlags usage, bool mipMapped)
{
    AllocatedImage newImage;
    newImage.imageFormat = format;
    newImage.imageExtent = imageExtent;

    VkImageCreateInfo imgInfo = vkinit::image_create_info(format, usage, imageExtent);
    if(mipMapped)
    {
        imgInfo.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(imageExtent.width, imageExtent.height)))) + 1;
    }

    // Always allocate images on dedicated GPU memory
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    allocInfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Allocate and create the image
    VK_CHECK(vmaCreateImage(renderer->vmaAllocator, &imgInfo, &allocInfo, &newImage.image, &newImage.allocation, nullptr));

    // Defaulting to the color aspect unless depth format is given
    VkImageAspectFlags aspectFlag = VK_IMAGE_ASPECT_COLOR_BIT; 
    if(format == VK_FORMAT_D32_SFLOAT) // if the format is the depth format
    {
        aspectFlag = VK_IMAGE_ASPECT_DEPTH_BIT;
    }

    // Create the image-view for the image
    VkImageViewCreateInfo viewInfo = vkinit::imageview_create_info(format, newImage.image, aspectFlag);
    viewInfo.subresourceRange.levelCount = imgInfo.mipLevels;

    VK_CHECK(vkCreateImageView(renderer->device, &viewInfo, nullptr, &newImage.imageView));

    return newImage;
}

AllocatedImage SK::VkRendererBackend::createImage(Renderer* renderer, void* data, VkExtent3D imageExtent, VkFormat format, VkImageUsageFlags usage, bool mipMapped)
{
    // Hardcoding the textures to be RGBA 8 bit format. This should be sufficient as most of the textures are in that format.
    size_t dataSize = imageExtent.depth * imageExtent.width * imageExtent.height * 4;
    AllocatedBuffer uploadBuffer = createBuffer(renderer, dataSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    memcpy(uploadBuffer.allocInfo.pMappedData, data, dataSize);

    // aside from the original usage also allow copying data into and from it.
    AllocatedImage newImage = createImage(renderer, imageExtent, format, usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, mipMapped);

    // Perform a buffer to image copy.
    immediateSubmit(renderer, [&](VkCommandBuffer cmd) {
        vkutil::transitionImage(cmd, newImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        VkBufferImageCopy copyRegion{};
        copyRegion.bufferOffset = 0;
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;

        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.mipLevel = 0;
        copyRegion.imageSubresource.baseArrayLayer = 0;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageExtent = imageExtent;

        // copy buffer to image
        vkCmdCopyBufferToImage(cmd, uploadBuffer.buffer, newImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

        if(mipMapped)
        {
            vkutil::generateMipmaps(cmd, newImage.image, VkExtent2D{ newImage.imageExtent.width, newImage.imageExtent.height });
        }
        else
        {
            vkutil::transitionImage(cmd, newImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }

    });

    destroyBuffer(renderer, uploadBuffer);

    return newImage;
}

void SK::VkRendererBackend::destroyImage(Renderer* renderer, const AllocatedImage& img)
{
    vkDestroyImageView(renderer->device, img.imageView, nullptr);
    vmaDestroyImage(renderer->vmaAllocator, img.image, img.allocation);
}

GPUMeshBuffers SK::VkRendererBackend::uploadMesh(Renderer* renderer, std::span<Vertex> vertices, std::span<uint32_t> indices)
{
    const size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
    const size_t indexBufferSize = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers meshBuffers;

    // Create the vertex buffer and fetch the device address of it
    meshBuffers.vertexBuffer = createBuffer(renderer, vertexBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    VkBufferDeviceAddressInfo deviceAddressInfo{ .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = meshBuffers.vertexBuffer.buffer };
    meshBuffers.vertexBufferAddress = vkGetBufferDeviceAddress(renderer->device, &deviceAddressInfo);

    // Create the index buffer
    meshBuffers.indexBuffer = createBuffer(renderer, indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    AllocatedBuffer staging = createBuffer(renderer, vertexBufferSize + indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    void* data = staging.allocation->GetMappedData();

    // Copy Vertex Buffer
    memcpy(data, vertices.data(), vertexBufferSize);
    // Copy Index Buffer
    memcpy((char*)data + vertexBufferSize, indices.data(), indexBufferSize);

    immediateSubmit(renderer, [&](VkCommandBuffer cmd) {
        VkBufferCopy vertexCopy{};
        vertexCopy.dstOffset = 0;
        vertexCopy.srcOffset = 0;
        vertexCopy.size = vertexBufferSize;

        vkCmdCopyBuffer(cmd, staging.buffer, meshBuffers.vertexBuffer.buffer, 1, &vertexCopy);

        VkBufferCopy indexCopy{};
        indexCopy.dstOffset = 0;
        indexCopy.srcOffset = vertexBufferSize;
        indexCopy.size = indexBufferSize;

        vkCmdCopyBuffer(cmd, staging.buffer, meshBuffers.indexBuffer.buffer, 1, &indexCopy);
    });

    destroyBuffer(renderer, staging);

    return meshBuffers;
}

/*
    Both update and bind scene buffer functions must be called after the frame fence waits as it will be guaranteed that the frame is done being used by GPU. Otherwise, the data can be corrupted. 
    (calling in drawMain() will suffice)
*/
void SK::VkRendererBackend::updateSceneBuffer(Renderer* renderer, const GPUSceneData& sceneData)
{
    // Update the scene buffer
    GPUSceneData* pGpuSceneDataBuffer = (GPUSceneData*)renderer->gpuSceneDataBuffer[renderer->frameNumber % FRAME_OVERLAP].allocation->GetMappedData();
    *pGpuSceneDataBuffer = sceneData;
}

VkDescriptorSet SK::VkRendererBackend::fetchCurrentSceneBufferDescriptorSet(Renderer* renderer)
{
    return renderer->gpuSceneDescriptorSet[renderer->frameNumber % FRAME_OVERLAP];
}

void SK::VkRendererBackend::setViewport(Renderer* renderer, VkCommandBuffer cmd)
{
    VkViewport viewport = {};
    viewport.x = 0;
    viewport.y = 0;
    viewport.width = renderer->drawExtent.width;
    viewport.height = renderer->drawExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);
}

void SK::VkRendererBackend::setScissor(Renderer* renderer, VkCommandBuffer cmd)
{
    VkRect2D scissor = {};
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent.width = renderer->drawExtent.width;
    scissor.extent.height = renderer->drawExtent.height;
    vkCmdSetScissor(cmd, 0, 1, &scissor);
}

SK::VkRendererBackend::FrameData& SK::VkRendererBackend::fetchCurrentFrameData(Renderer* renderer)
{
    return renderer->frames[renderer->frameNumber % FRAME_OVERLAP];
}

void SK::VkRendererBackend::registerOverlayPass(Renderer* renderer, OverlayPass pass)
{
    renderer->overlayPasses.push_back(pass);
}

VkShaderModule SK::VkRendererBackend::getOrLoadShader(Renderer* renderer, const char* path)
{
    size_t hash = std::hash<std::string>{}(path);

    auto it = renderer->shaderCache.find(hash);
    if(it != renderer->shaderCache.end())
    {
        return it->second;
    }

    VkShaderModule shaderModule;
    if(!vkutil::loadShaderModule(renderer->device, path, &shaderModule))
    {
        return VK_NULL_HANDLE;
    }

    renderer->shaderCache[hash] = shaderModule;
    return shaderModule;
}
    
void SK::VkRendererBackend::clearShaderCache(Renderer* renderer)
{
    for(auto& [k, s] : renderer->shaderCache)
    {
        vkDestroyShaderModule(renderer->device, s, nullptr);
    }
    renderer->shaderCache.clear();
}

VkPipeline SK::VkRendererBackend::getOrCreatePipeline(Renderer* renderer, const PipelineKey& key)
{
    size_t hash = hashPipelineKey(key);

    auto it = renderer->pipelineCache.find(hash);
    if(it != renderer->pipelineCache.end())
    {
        return it->second;
    }

    PipelineBuilder builder;
    builder.clear();

    builder.setShaders(
        renderer->shaderCache[key.vertShader],
        renderer->shaderCache[key.fragShader]
    );

    builder.setInputTopology(key.topology);
    builder.setPolygonMode(key.polygonMode);
    builder.setCullMode(key.cullMode, key.frontFace);
    // Hardcoding for now
    builder.setMultiSamplingNone();

    if(key.blending)
    {
        builder.enableBlendingAdditive();
    }
    else
    {
        builder.disableBlending();
    }

    if(key.depthTest)
    {
        builder.enableDepthTest(key.depthWrite, key.depthCompare);
    }
    else
    {
        builder.disableDepthTest();
    }

    builder.setColorAttachmentFormat(key.colorFormat);
    builder.setDepthFormat(key.depthFormat);

    builder.pipelineLayout = key.layout;

    VkPipeline pipeline = builder.buildPipeline(renderer->device);

    renderer->pipelineCache[hash] = pipeline;
    return pipeline;
}

void SK::VkRendererBackend::clearPipelineCache(Renderer* renderer)
{
    for(auto& [k, p] : renderer->pipelineCache)
    {
        vkDestroyPipeline(renderer->device, p, nullptr);
    }
    renderer->pipelineCache.clear();
}

void SK::VkRendererBackend::m_initVulkan(Renderer* renderer)
{
    vkb::InstanceBuilder builder;

    // Create the Vulkan instance with basic debug features.
    auto instRet = builder.set_app_name("Vulkan Engine")
        .request_validation_layers(useValidationLayers)
        .use_default_debug_messenger()
        .require_api_version(1, 3, 0)
        .build();

    vkb::Instance vkbInstance = instRet.value();

    // Grab the instance
    renderer->instance = vkbInstance.instance;
    renderer->debugMessenger = vkbInstance.debug_messenger;

    SDL_Vulkan_CreateSurface(renderer->window, renderer->instance, &renderer->surface);

    // Vulkan 1.3 features
    VkPhysicalDeviceVulkan13Features features13{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    features13.dynamicRendering = true;
    features13.synchronization2 = true;

    // Vulkan 1.2 features
    VkPhysicalDeviceVulkan12Features features12{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    features12.bufferDeviceAddress = true;
    features12.descriptorIndexing = true;

    // Use vkbootstrap to select a gpu with Vulkan 1.3 and necessary features
    vkb::PhysicalDeviceSelector selector{vkbInstance};
    vkb::PhysicalDevice physicalDevice = selector
        .set_minimum_version(1, 3)
        .set_required_features_13(features13)
        .set_required_features_12(features12)
        .set_surface(renderer->surface)
        .select()
        .value();

    // Create the final Vulkan device
    vkb::DeviceBuilder deviceBuilder{physicalDevice};
    vkb::Device vkbDevice = deviceBuilder.build().value();

    // Get the VKDevice handle used in the rest of the Vulkan application
    renderer->device = vkbDevice.device;
    renderer->chosenGPU = vkbDevice.physical_device;
    // Get the Graphics Queue
    renderer->graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    renderer->graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    // Initialize the memory allocator
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = renderer->chosenGPU;
    allocatorInfo.device = renderer->device;
    allocatorInfo.instance = renderer->instance;
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&allocatorInfo, &renderer->vmaAllocator);

    renderer->mainDeletionQueue.pushFunction([=](){
        vmaDestroyAllocator(renderer->vmaAllocator);
    });
}

void SK::VkRendererBackend::m_initSwapchain(Renderer* renderer)
{
    m_createSwapchain(renderer, renderer->windowExtent.width, renderer->windowExtent.height);

    // draw image size will match the window
    VkExtent3D drawImageExtent = {
        renderer->windowExtent.width,
        renderer->windowExtent.height,
        1
    };

    renderer->drawImage.imageExtent = drawImageExtent;

    // Hardcoding the draw format to 16 bit float
    renderer->drawImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

    VkImageUsageFlags drawImageUsageFlags{};
    drawImageUsageFlags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    drawImageUsageFlags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    drawImageUsageFlags |= VK_IMAGE_USAGE_STORAGE_BIT;
    drawImageUsageFlags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    VkImageCreateInfo drawImageInfo = vkinit::image_create_info(renderer->drawImage.imageFormat, drawImageUsageFlags, renderer->drawImage.imageExtent);

    // For the draw image, we want to allocate it from the gpu local memory
    VmaAllocationCreateInfo imageAllocInfo = {};
    imageAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    imageAllocInfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Allocate and create the image
    vmaCreateImage(renderer->vmaAllocator, &drawImageInfo, &imageAllocInfo, &renderer->drawImage.image, &renderer->drawImage.allocation, nullptr);

    // Build an image-view for the draw image to use for rendering
    VkImageViewCreateInfo drawImageViewInfo = vkinit::imageview_create_info(renderer->drawImage.imageFormat, renderer->drawImage.image, VK_IMAGE_ASPECT_COLOR_BIT);

    VK_CHECK(vkCreateImageView(renderer->device, &drawImageViewInfo, nullptr, &renderer->drawImage.imageView));

    // Initialize the depth image
    renderer->depthImage.imageFormat = VK_FORMAT_D32_SFLOAT; // one-component, 32-bit signed floating-point format that has 32 bits in the depth component
    renderer->depthImage.imageExtent = drawImageExtent;
    VkImageUsageFlags depthImageUsages{};
    depthImageUsages |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    VkImageCreateInfo depthImageInfo = vkinit::image_create_info(renderer->depthImage.imageFormat, depthImageUsages, renderer->depthImage.imageExtent);
    vmaCreateImage(renderer->vmaAllocator, &depthImageInfo, &imageAllocInfo, &renderer->depthImage.image, &renderer->depthImage.allocation, nullptr);
    VkImageViewCreateInfo depthViewInfo = vkinit::imageview_create_info(renderer->depthImage.imageFormat, renderer->depthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT);
    VK_CHECK(vkCreateImageView(renderer->device, &depthViewInfo, nullptr, &renderer->depthImage.imageView));

    // Add the resources to the deletion queue
    renderer->mainDeletionQueue.pushFunction([=](){
        // Destroy the Draw Image
        vmaDestroyImage(renderer->vmaAllocator, renderer->drawImage.image, renderer->drawImage.allocation);
        vkDestroyImageView(renderer->device, renderer->drawImage.imageView, nullptr);
        // Destroy the Depth Image
        vmaDestroyImage(renderer->vmaAllocator, renderer->depthImage.image, renderer->depthImage.allocation);
        vkDestroyImageView(renderer->device, renderer->depthImage.imageView, nullptr);
    });
}

void SK::VkRendererBackend::m_initCommands(Renderer* renderer)
{
    // Create the command pool and allow for resetting of individual command buffers
    VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(renderer->graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

    for(int i = 0; i < FRAME_OVERLAP; ++i)
    {
        VK_CHECK(vkCreateCommandPool(renderer->device, &commandPoolInfo, nullptr, &renderer->frames[i].commandPool));
        // Allocate the default command buffer that will be used for rendering
        VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(renderer->frames[i].commandPool, 1);
        VK_CHECK(vkAllocateCommandBuffers(renderer->device, &cmdAllocInfo, &renderer->frames[i].mainCommandBuffer));
    }

    // Immediate commands
    VK_CHECK(vkCreateCommandPool(renderer->device, &commandPoolInfo, nullptr, &renderer->immediateCommandPool));

    // Allocate a command buffer for immediate submits
    VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(renderer->immediateCommandPool, 1);

    VK_CHECK(vkAllocateCommandBuffers(renderer->device, &cmdAllocInfo, &renderer->immediateCommandBuffer));

    renderer->mainDeletionQueue.pushFunction([=](){
        vkDestroyCommandPool(renderer->device, renderer->immediateCommandPool, nullptr);
    });
}

void SK::VkRendererBackend::m_initSyncStructures(Renderer* renderer)
{
    //create syncronization structures
    //one fence to control when the gpu has finished rendering the frame,
    //and 2 semaphores to syncronize rendering with swapchain
    //we want the fence to start signalled so we can wait on it on the first frame
    VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

    for(int i = 0; i < FRAME_OVERLAP; ++i)
    {
        VK_CHECK(vkCreateFence(renderer->device, &fenceCreateInfo, nullptr, &renderer->frames[i].renderFence));

        VK_CHECK(vkCreateSemaphore(renderer->device, &semaphoreCreateInfo, nullptr, &renderer->frames[i].swapchainSemaphore));
        VK_CHECK(vkCreateSemaphore(renderer->device, &semaphoreCreateInfo, nullptr, &renderer->frames[i].renderSemaphore));
    }

    // Fence for the immediate command buffers
    VK_CHECK(vkCreateFence(renderer->device, &fenceCreateInfo, nullptr, &renderer->immeadiateFence));
    renderer->mainDeletionQueue.pushFunction([=](){
        vkDestroyFence(renderer->device, renderer->immeadiateFence, nullptr);
    });
}

void SK::VkRendererBackend::m_createSwapchain(Renderer* renderer, uint32_t width, uint32_t height)
{
    vkb::SwapchainBuilder swapchainBuilder{ renderer->chosenGPU, renderer->device, renderer->surface};

    renderer->swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;

    vkb::Swapchain vkbSwapchain = swapchainBuilder
        .set_desired_format(VkSurfaceFormatKHR{.format = renderer->swapchainImageFormat, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(width, height)
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .build()
        .value();

    
    renderer->swapchainExtent = vkbSwapchain.extent;
    // Store the swapchain and its related images
    renderer->swapchain = vkbSwapchain.swapchain;
    renderer->swapchainImages = vkbSwapchain.get_images().value();
    renderer->swapchainImageViews = vkbSwapchain.get_image_views().value();
}

void SK::VkRendererBackend::m_destroySwapchain(Renderer* renderer)
{
    // Deleting the swapchain deletes the images it holds internally.
    vkDestroySwapchainKHR(renderer->device, renderer->swapchain, nullptr);

    // Destroy the swapchain resources
    for(int i = 0; i < renderer->swapchainImageViews.size(); ++i)
    {
        vkDestroyImageView(renderer->device, renderer->swapchainImageViews[i], nullptr);
    }

    renderer->swapchainImages.clear();
    renderer->swapchainImageViews.clear();
}

void SK::VkRendererBackend::m_resizeSwapchain(Renderer* renderer)
{
    // Don't change the images and views while the gpu is still handling them
    vkDeviceWaitIdle(renderer->device);

    m_destroySwapchain(renderer);

    int w, h;
    SDL_GetWindowSize(renderer->window, &w, &h);
    renderer->windowExtent.width = w;
    renderer->windowExtent.height = h;

    m_createSwapchain(renderer, renderer->windowExtent.width, renderer->windowExtent.height);

    renderer->resizeRequested = false;
}

void SK::VkRendererBackend::m_initDescriptors(Renderer* renderer)
{
    // Create the global growable descriptor allocator 
    std::vector<DescriptorAllocatorGrowable::PoolSize> sizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 }
    };

    renderer->globalDescriptorAllocator.init(renderer->device, 10, sizes);
    
    // The descriptor set layout for the main draw image
    {
        DescriptorLayoutBuilder builder;
        builder.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        renderer->drawImageDescriptorSetLayout = builder.build(renderer->device, VK_SHADER_STAGE_COMPUTE_BIT);
    }

    // The descriptor set layout for single texture display
    {
        DescriptorLayoutBuilder builder;
        builder.addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        renderer->displayTextureDescriptorSetLayout = builder.build(renderer->device, VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    // Descriptor set layout for the scene data
    {
        DescriptorLayoutBuilder builder;
        builder.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        renderer->gpuSceneDataDescriptorLayout = builder.build(renderer->device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    // Allocate a descriptor set for the draw image
    renderer->drawImageDescriptorSet = renderer->globalDescriptorAllocator.allocate(renderer->device, renderer->drawImageDescriptorSetLayout);

    {
        DescriptorWriter writer;
        writer.writeImage(0, renderer->drawImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        writer.updateSet(renderer->device, renderer->drawImageDescriptorSet);
    }

    // Add the descriptor allocator and layout destructors to the deletion queue
    renderer->mainDeletionQueue.pushFunction([=](){
        renderer->globalDescriptorAllocator.destroyPools(renderer->device);

        vkDestroyDescriptorSetLayout(renderer->device, renderer->drawImageDescriptorSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(renderer->device, renderer->displayTextureDescriptorSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(renderer->device, renderer->gpuSceneDataDescriptorLayout, nullptr);
    });

    // Init the per-frame descriptor allocators
    for(int i = 0; i < FRAME_OVERLAP; ++i)
    {
        std::vector<DescriptorAllocatorGrowable::PoolSize> framePoolSizes = {
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 },
        };

        renderer->frames[i].frameDescriptorAllocator = DescriptorAllocatorGrowable{};
        renderer->frames[i].frameDescriptorAllocator.init(renderer->device, 1000, framePoolSizes);

        // Pools in the frame descriptor allocators must be destroyed with the renderer shutdown (not with frame shutdown)
        renderer->mainDeletionQueue.pushFunction([=]() {
            renderer->frames[i].frameDescriptorAllocator.destroyPools(renderer->device);
        });
    }
}

void SK::VkRendererBackend::m_initPasses(Renderer* renderer)
{
    GLTFMetallicPass::Init(renderer);
}

void SK::VkRendererBackend::m_clearPassResources(Renderer* renderer)
{
    GLTFMetallicPass::ClearResources(renderer);
}

void SK::VkRendererBackend::m_initMaterialLayouts(Renderer* renderer)
{
    GLTFMetallicRoughnessMaterial::BuildMaterialLayout(renderer);
}

void SK::VkRendererBackend::m_clearMaterialLayouts(Renderer* renderer)
{
    GLTFMetallicRoughnessMaterial::ClearMaterialLayout(renderer->device);
}

void SK::VkRendererBackend::m_initDefaultData(Renderer* renderer)
{
    // Default textures
    // 3 default textures 1 pixel each
    uint32_t white = glm::packUnorm4x8(glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
    renderer->whiteImage = createImage(renderer, (void*)&white, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t grey = glm::packUnorm4x8(glm::vec4(0.66f, 0.66f, 0.66f, 1.0f));
    renderer->greyImage = createImage(renderer, (void*)&grey, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t black = glm::packUnorm4x8(glm::vec4(0.0f, 0.0f, 0.0f, 0.0f));
    renderer->blackImage =createImage(renderer, (void*)&black, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    //checkerboard image
    uint32_t magenta = glm::packUnorm4x8(glm::vec4(1, 0, 1, 1));
    std::array<uint32_t, 16 * 16 > pixels; //for 16x16 checkerboard texture
    for(int y = 0; y < 16; ++y) 
    {
        for(int x = 0; x < 16; ++x) 
        {
            pixels[y * 16 + x] = ((x % 2) ^ (y % 2)) ? magenta : black;
        }
    }

    renderer->errorCheckerboardImage = createImage(renderer, pixels.data(), VkExtent3D{ 16, 16, 1 }, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    // Default samplers
    VkSamplerCreateInfo samplerInfo = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };

    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    vkCreateSampler(renderer->device, &samplerInfo, nullptr, &renderer->defaultSamplerNearest);

    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    vkCreateSampler(renderer->device, &samplerInfo, nullptr, &renderer->defaultSamplerLinear);

    renderer->mainDeletionQueue.pushFunction([=]() {
        destroyImage(renderer, renderer->whiteImage);
        destroyImage(renderer, renderer->greyImage);
        destroyImage(renderer, renderer->blackImage);
        destroyImage(renderer, renderer->errorCheckerboardImage);

        vkDestroySampler(renderer->device, renderer->defaultSamplerNearest, nullptr);
        vkDestroySampler(renderer->device, renderer->defaultSamplerLinear, nullptr);
    });

    // Default material data
    GLTFMetallicRoughnessMaterial::MaterialResources defaultMaterialResources;
    defaultMaterialResources.colorImage = renderer->whiteImage;
    defaultMaterialResources.colorSampler = renderer->defaultSamplerLinear;
    defaultMaterialResources.metalRoughnessImage = renderer->whiteImage;
    defaultMaterialResources.metalRoughnessSampler = renderer->defaultSamplerLinear;
    
    AllocatedBuffer materialConstantsBuffer = createBuffer(renderer, sizeof(GLTFMetallicRoughnessMaterial::MaterialConstants), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    // Write the buffer
    GLTFMetallicRoughnessMaterial::MaterialConstants* pMaterialConstantsBuffer = static_cast<GLTFMetallicRoughnessMaterial::MaterialConstants*>(materialConstantsBuffer.allocation->GetMappedData());
    pMaterialConstantsBuffer->colorFactors = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    pMaterialConstantsBuffer->metalRoughnessFactors = glm::vec4(1.0f, 0.5f, 0.0f, 0.0f);

    renderer->mainDeletionQueue.pushFunction([=]() {
        destroyBuffer(renderer, materialConstantsBuffer);
    });

    defaultMaterialResources.dataBuffer = materialConstantsBuffer.buffer;
    defaultMaterialResources.dataBufferOffset = 0;

    renderer->defaultMaterialInstance = GLTFMetallicRoughnessMaterial::CreateInstance(renderer->device, MaterialPass::Opaque, defaultMaterialResources, renderer->globalDescriptorAllocator);
}

void SK::VkRendererBackend::m_initGlobalSceneBuffer(Renderer* renderer)
{
    for(int i = 0; i < FRAME_OVERLAP; ++i)
    {
        // Allocate a new uniform buffer for scene data (allocating on VRAM that CPU can write to directly. It is limited but it is perfect for allocating reasonable amounts that are dynamic)
        renderer->gpuSceneDataBuffer[i] = createBuffer(renderer, sizeof(GPUSceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        renderer->mainDeletionQueue.pushFunction([=]() {
            destroyBuffer(renderer, renderer->gpuSceneDataBuffer[i]);
        });

        // Create a descriptor set for the uniform data
        renderer->gpuSceneDescriptorSet[i] = renderer->globalDescriptorAllocator.allocate(renderer->device, renderer->gpuSceneDataDescriptorLayout);
        DescriptorWriter writer;
        writer.writeBuffer(0, renderer->gpuSceneDataBuffer[i].buffer, sizeof(GPUSceneData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        writer.updateSet(renderer->device, renderer->gpuSceneDescriptorSet[i]);
    }
}
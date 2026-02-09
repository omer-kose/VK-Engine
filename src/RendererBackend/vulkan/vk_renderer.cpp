#include "vk_renderer.h"

#include <chrono>
#include <thread>
#include <array>

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

static size_t hashPipelineLayoutKey(const SK::VkRendererBackend::PipelineLayoutKey& k)
{
    size_t h = 0;

    auto hc = [&](auto v)
    {
        std::hash<uint64_t> hasher;
        h ^= hasher((uint64_t)v) + 0x9e3779b9 + (h << 6) + (h >> 2);
    };
    
    for(auto l : k.setLayouts)
    {
        hc(l);
    }

    for(auto& pc : k.pushConstantRanges)
    {
        hc(pc.offset);
        hc(pc.size);
        hc(pc.stageFlags);
    }

    return h;
}

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


void SK::VkRendererBackend::init(State* vkRendererBackend, struct SDL_Window* window, uint32_t windowWidth, uint32_t windowHeight)
{
    // only one vkRendererBackend initialization is allowed with the application.
    assert(vkRendererBackend->isInitialized == false);

    // Store window data coming from the App for later use
    vkRendererBackend->window = window;
    vkRendererBackend->windowExtent = VkExtent2D{ windowWidth, windowHeight };

    // Vulkan Bootstrapping
    m_initVulkan(vkRendererBackend);
    m_initSwapchain(vkRendererBackend);
    m_initCommands(vkRendererBackend);
    m_initSyncStructures(vkRendererBackend);

    createDrawAndDepthImages(vkRendererBackend);

    m_initDescriptors(vkRendererBackend);

    m_initMaterialLayouts(vkRendererBackend);

    m_initDefaultData(vkRendererBackend);

    m_initGlobalSceneBuffer(vkRendererBackend);

    // everything went fine
    vkRendererBackend->isInitialized = true;
}

void SK::VkRendererBackend::shutdown(State* vkRendererBackend)
{
    if(vkRendererBackend->isInitialized) 
    {
        for(int i = 0; i < FRAME_OVERLAP; ++i)
        {
            // Destroy sync objects
            vkDestroyFence(vkRendererBackend->device, vkRendererBackend->frames[i].renderFence, nullptr);
            vkDestroySemaphore(vkRendererBackend->device, vkRendererBackend->frames[i].swapchainSemaphore, nullptr);
            vkDestroySemaphore(vkRendererBackend->device, vkRendererBackend->frames[i].renderSemaphore, nullptr);

            // It’s not possible to individually destroy VkCommandBuffer, destroying their parent pool will destroy all of the command buffers allocated from it.
            vkDestroyCommandPool(vkRendererBackend->device, vkRendererBackend->frames[i].commandPool, nullptr);

            vkRendererBackend->frames[i].deletionQueue.flush();
        }

        // Clear out the caches
        clearShaderCache(vkRendererBackend);
        clearPipelineLayoutCache(vkRendererBackend);
        clearPipelineCache(vkRendererBackend);

        m_clearMaterialLayouts(vkRendererBackend);

        destroyDrawAndDepthImages(vkRendererBackend);

        // destroying the vma allocator is also inside the mainDeletionQueue, so any resource allocation must be freed before flushing the queue
        vkRendererBackend->mainDeletionQueue.flush();

        m_destroySwapchain(vkRendererBackend);

        vkDestroyDevice(vkRendererBackend->device, nullptr);
        vkDestroySurfaceKHR(vkRendererBackend->instance, vkRendererBackend->surface, nullptr);

        vkb::destroy_debug_utils_messenger(vkRendererBackend->instance, vkRendererBackend->debugMessenger);
        vkDestroyInstance(vkRendererBackend->instance, nullptr);

        // Nullify non-owning pointers
        vkRendererBackend->window = nullptr;
        
        vkRendererBackend->isInitialized = false;
    }
}

/*
    Prepares/synchronizes internal logic and prepares the frame to be drawn.

    Returns true if the frame begun successfully. In the cases like swapchain resize, it returns false.
*/
bool SK::VkRendererBackend::beginFrame(State* vkRendererBackend)
{
    FrameData& currentFrame = getCurrentFrameData(vkRendererBackend);
    // Wait until the GPU has finished rendering the last frame of the same modularity (0->1->2->3  wait on 2 for 0 and wait on 3 for 1 and so on)
    VK_CHECK(vkWaitForFences(vkRendererBackend->device, 1, &currentFrame.renderFence, true, 1000000000));

    currentFrame.deletionQueue.flush();
    currentFrame.frameDescriptorAllocator.clearPools(vkRendererBackend->device);

    // To be able to use the same fence it must be reset after use
    VK_CHECK(vkResetFences(vkRendererBackend->device, 1, &currentFrame.renderFence));

    // Request an available image from the swapchain. swapchainSemaphore is signaled once it has finished presenting the image so it can be used again.
    // More detailed description of how vkAcquireNextImageKHR works: https://stackoverflow.com/questions/60419749/why-does-vkacquirenextimagekhr-never-block-my-thread
    uint32_t swapchainImageIndex;
    VkResult acquireResult = vkAcquireNextImageKHR(vkRendererBackend->device, vkRendererBackend->swapchain, 1000000000, currentFrame.swapchainSemaphore, nullptr, &swapchainImageIndex);
    if(acquireResult == VK_ERROR_OUT_OF_DATE_KHR)
    {
        vkRendererBackend->windowResizeRequested = true;
        return false;
    }

    // Set the extent of the image that we are going to draw onto
    vkRendererBackend->drawExtent.width = std::min(vkRendererBackend->drawImage.imageExtent.width, vkRendererBackend->swapchainExtent.width) * vkRendererBackend->renderScale;
    vkRendererBackend->drawExtent.height = std::min(vkRendererBackend->drawImage.imageExtent.height, vkRendererBackend->swapchainExtent.height) * vkRendererBackend->renderScale;

    // Vulkan handles are just a 64 bit handles/pointers, so its fine to copy them around, but remember that their actual data is handled by vulkan itself.
    VkCommandBuffer cmd = currentFrame.mainCommandBuffer;

    // Now we are sure that command is executed, we can safely reset it and begin recording again
    VK_CHECK(vkResetCommandBuffer(cmd, 0));

    // Begin the command buffer recording. We will submit this command buffer exactly once, so we let Vulkan know that
    VkCommandBufferBeginInfo cmdBeginInfo = SK::VkInit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    // Start the command buffer recording
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    // Transition depth image to optimal depth layout
    SK::VkUtil::transitionImage(cmd, vkRendererBackend->depthImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    // Transition draw image to optimal rendering layout
    SK::VkUtil::transitionImage(cmd, vkRendererBackend->drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // Frame has begun successfully, fill in per-frame transient state in the renderer backend state
    vkRendererBackend->currentCmdBuffer = currentFrame.mainCommandBuffer;
    vkRendererBackend->currentSwapchainImageIndex = swapchainImageIndex;

    return true;
}

void SK::VkRendererBackend::drawOverlays(State* vkRendererBackend)
{
    VkCommandBuffer cmd = vkRendererBackend->currentCmdBuffer;
    uint32_t swapchainImageIndex = vkRendererBackend->currentSwapchainImageIndex;

    SK::VkUtil::transitionImage(cmd, vkRendererBackend->drawImage.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    SK::VkUtil::transitionImage(cmd, vkRendererBackend->swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // Execute a copy operation from the draw image into the swapchain image
    SK::VkUtil::copyImageToImage(cmd, vkRendererBackend->drawImage.image, vkRendererBackend->swapchainImages[swapchainImageIndex], vkRendererBackend->drawExtent, vkRendererBackend->swapchainExtent);

    // After drawing, we need to draw overlays on top of the swapchain image, so transition the swapchain image into optimal drawing layout
    SK::VkUtil::transitionImage(cmd, vkRendererBackend->swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // Execute overlay passes
    for(auto& pass : vkRendererBackend->overlayPasses)
    {
        PassContext ctx = { cmd, vkRendererBackend->swapchainImageViews[swapchainImageIndex], vkRendererBackend->swapchainExtent, vkRendererBackend };
        pass.draw(&ctx);
    }
}

void SK::VkRendererBackend::endFrame(State* vkRendererBackend)
{
    FrameData& currentFrame = getCurrentFrameData(vkRendererBackend);

    VkCommandBuffer cmd = vkRendererBackend->currentCmdBuffer;
    uint32_t swapchainImageIndex = vkRendererBackend->currentSwapchainImageIndex;

    // Transition swapchain image into the presentation layout
    SK::VkUtil::transitionImage(cmd, vkRendererBackend->swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    // Finalize the command buffer
    VK_CHECK(vkEndCommandBuffer(cmd));

    // Prepare the submission
    // We will wait on the swapchainSemaphore before executing the commands as that semaphore is signaled once swapchain is done presenting that image
    // We will signal renderSemaphore to signal that rendering has finished
    VkCommandBufferSubmitInfo cmdSubmitInfo = SK::VkInit::command_buffer_submit_info(cmd);
    VkSemaphoreSubmitInfo waitInfo = SK::VkInit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, currentFrame.swapchainSemaphore);
    VkSemaphoreSubmitInfo signalInfo = SK::VkInit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, currentFrame.renderSemaphore);
    VkSubmitInfo2 submit = SK::VkInit::submit_info(&cmdSubmitInfo, &signalInfo, &waitInfo);

    // Submit command buffer to the queue and execute it
    // renderFence will be signaled once the submitted command buffer has completed execution.
    VK_CHECK(vkQueueSubmit2(vkRendererBackend->graphicsQueue, 1, &submit, currentFrame.renderFence));

    // Prepare the presentation
    // We will wait on the renderSemaphore so that it will be guaranteed that the rendering has been finished and the swapchain image is ready to be presented
    VkPresentInfoKHR presentInfo = { .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, .pNext = nullptr };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &vkRendererBackend->swapchain;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &currentFrame.renderSemaphore;
    presentInfo.pImageIndices = &swapchainImageIndex;

    VkResult presentResult = vkQueuePresentKHR(vkRendererBackend->graphicsQueue, &presentInfo);
    if(presentResult == VK_ERROR_OUT_OF_DATE_KHR)
    {
        vkRendererBackend->windowResizeRequested = true;
    }

    // Increase the number of frames drawn
    ++vkRendererBackend->frameNumber;
}

void SK::VkRendererBackend::immediateSubmit(State* vkRendererBackend, std::function<void(VkCommandBuffer cmd)>&& function)
{
    // Before starting submitting and waiting on the fence reset them
    VK_CHECK(vkResetFences(vkRendererBackend->device, 1, &vkRendererBackend->immeadiateFence));
    VK_CHECK(vkResetCommandBuffer(vkRendererBackend->immediateCommandBuffer, 0));
    // Prepare the immediate command buffer for executing function given as the param
    VkCommandBuffer cmd = vkRendererBackend->immediateCommandBuffer;
    VkCommandBufferBeginInfo cmdBeginInfo = SK::VkInit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));
    function(cmd);
    VK_CHECK(vkEndCommandBuffer(cmd));

    // Submit
    VkCommandBufferSubmitInfo cmdSubmitInfo = SK::VkInit::command_buffer_submit_info(cmd);
    VkSubmitInfo2 submitInfo = SK::VkInit::submit_info(&cmdSubmitInfo, nullptr, nullptr);
    VK_CHECK(vkQueueSubmit2(vkRendererBackend->graphicsQueue, 1, &submitInfo, vkRendererBackend->immeadiateFence));

    // Wait on the fence until the command buffer finished executing
    VK_CHECK(vkWaitForFences(vkRendererBackend->device, 1, &vkRendererBackend->immeadiateFence, true, 9999999999));
}

AllocatedBuffer SK::VkRendererBackend::createBuffer(State* vkRendererBackend, size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage)
{
    // Allocate buffer
    VkBufferCreateInfo bufferInfo = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = nullptr };
    bufferInfo.size = allocSize;
    bufferInfo.usage = usage;

    VmaAllocationCreateInfo vmaAllocInfo = {};
    vmaAllocInfo.usage = memoryUsage;
    vmaAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    AllocatedBuffer newBuffer;

    VK_CHECK(vmaCreateBuffer(vkRendererBackend->vmaAllocator, &bufferInfo, &vmaAllocInfo, &newBuffer.buffer, &newBuffer.allocation, &newBuffer.allocInfo));

    return newBuffer;
}

void SK::VkRendererBackend::destroyBuffer(State* vkRendererBackend, const AllocatedBuffer& buffer)
{
    vmaDestroyBuffer(vkRendererBackend->vmaAllocator, buffer.buffer, buffer.allocation);
}

AllocatedImage SK::VkRendererBackend::createImage(State* vkRendererBackend, VkExtent3D imageExtent, VkFormat format, VkImageUsageFlags usage, bool mipMapped)
{
    AllocatedImage newImage;
    newImage.imageFormat = format;
    newImage.imageExtent = imageExtent;

    VkImageCreateInfo imgInfo = SK::VkInit::image_create_info(format, usage, imageExtent);
    if(mipMapped)
    {
        imgInfo.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(imageExtent.width, imageExtent.height)))) + 1;
    }

    // Always allocate images on dedicated GPU memory
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    allocInfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Allocate and create the image
    VK_CHECK(vmaCreateImage(vkRendererBackend->vmaAllocator, &imgInfo, &allocInfo, &newImage.image, &newImage.allocation, nullptr));

    // Defaulting to the color aspect unless depth format is given
    VkImageAspectFlags aspectFlag = VK_IMAGE_ASPECT_COLOR_BIT; 
    if(format == VK_FORMAT_D32_SFLOAT) // if the format is the depth format
    {
        aspectFlag = VK_IMAGE_ASPECT_DEPTH_BIT;
    }

    // Create the image-view for the image
    VkImageViewCreateInfo viewInfo = SK::VkInit::imageview_create_info(format, newImage.image, aspectFlag);
    viewInfo.subresourceRange.levelCount = imgInfo.mipLevels;

    VK_CHECK(vkCreateImageView(vkRendererBackend->device, &viewInfo, nullptr, &newImage.imageView));

    return newImage;
}

AllocatedImage SK::VkRendererBackend::createImage(State* vkRendererBackend, void* data, VkExtent3D imageExtent, VkFormat format, VkImageUsageFlags usage, bool mipMapped)
{
    // Hardcoding the textures to be RGBA 8 bit format. This should be sufficient as most of the textures are in that format.
    size_t dataSize = imageExtent.depth * imageExtent.width * imageExtent.height * 4;
    AllocatedBuffer uploadBuffer = createBuffer(vkRendererBackend, dataSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    memcpy(uploadBuffer.allocInfo.pMappedData, data, dataSize);

    // aside from the original usage also allow copying data into and from it.
    AllocatedImage newImage = createImage(vkRendererBackend, imageExtent, format, usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, mipMapped);

    // Perform a buffer to image copy.
    immediateSubmit(vkRendererBackend, [&](VkCommandBuffer cmd) {
        SK::VkUtil::transitionImage(cmd, newImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

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
            SK::VkUtil::generateMipmaps(cmd, newImage.image, VkExtent2D{ newImage.imageExtent.width, newImage.imageExtent.height });
        }
        else
        {
            SK::VkUtil::transitionImage(cmd, newImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }

    });

    destroyBuffer(vkRendererBackend, uploadBuffer);

    return newImage;
}

void SK::VkRendererBackend::destroyImage(State* vkRendererBackend, const AllocatedImage& img)
{
    vkDestroyImageView(vkRendererBackend->device, img.imageView, nullptr);
    vmaDestroyImage(vkRendererBackend->vmaAllocator, img.image, img.allocation);
}

VkSampler SK::VkRendererBackend::createSampler(State* vkRendererBackend, VkFilter minFilter, VkFilter magFilter, VkSamplerMipmapMode mipmapMode, VkSamplerAddressMode addressMode)
{
    VkSamplerCreateInfo createInfo = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };

    createInfo.minFilter = minFilter;
    createInfo.magFilter = magFilter;
    createInfo.mipmapMode = mipmapMode;
    createInfo.addressModeU = addressMode;
    createInfo.addressModeV = addressMode;
    createInfo.addressModeW = addressMode;
    createInfo.minLod = 0.0f;
    createInfo.maxLod = 16.0f;
    createInfo.anisotropyEnable = mipmapMode == VK_SAMPLER_MIPMAP_MODE_LINEAR;
    createInfo.maxAnisotropy = mipmapMode == VK_SAMPLER_MIPMAP_MODE_LINEAR ? 4.0f : 1.0f;

    VkSampler sampler;
    VK_CHECK(vkCreateSampler(vkRendererBackend->device, &createInfo, 0, &sampler));
    return sampler;
}

GPUMeshBuffers SK::VkRendererBackend::uploadMesh(State* vkRendererBackend, std::span<Vertex> vertices, std::span<uint32_t> indices)
{
    const size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
    const size_t indexBufferSize = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers meshBuffers;

    // Create the vertex buffer and fetch the device address of it
    meshBuffers.vertexBuffer = createBuffer(vkRendererBackend, vertexBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    VkBufferDeviceAddressInfo deviceAddressInfo{ .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = meshBuffers.vertexBuffer.buffer };
    meshBuffers.vertexBufferAddress = vkGetBufferDeviceAddress(vkRendererBackend->device, &deviceAddressInfo);

    // Create the index buffer
    meshBuffers.indexBuffer = createBuffer(vkRendererBackend, indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    AllocatedBuffer staging = createBuffer(vkRendererBackend, vertexBufferSize + indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    void* data = staging.allocation->GetMappedData();

    // Copy Vertex Buffer
    memcpy(data, vertices.data(), vertexBufferSize);
    // Copy Index Buffer
    memcpy((char*)data + vertexBufferSize, indices.data(), indexBufferSize);

    immediateSubmit(vkRendererBackend, [&](VkCommandBuffer cmd) {
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

    destroyBuffer(vkRendererBackend, staging);

    return meshBuffers;
}

/*
    Both update and bind scene buffer functions must be called after the frame fence waits as it will be guaranteed that the frame is done being used by GPU. Otherwise, the data can be corrupted. 
    So, it can be safely called after frameBegin function
*/
void SK::VkRendererBackend::updateSceneBuffer(State* vkRendererBackend, const GPUSceneData& sceneData)
{
    // Update the scene buffer
    GPUSceneData* pGpuSceneDataBuffer = (GPUSceneData*)vkRendererBackend->gpuSceneDataBuffer[vkRendererBackend->frameNumber % FRAME_OVERLAP].allocation->GetMappedData();
    *pGpuSceneDataBuffer = sceneData;
}

VkDescriptorSet SK::VkRendererBackend::fetchCurrentSceneBufferDescriptorSet(State* vkRendererBackend)
{
    return vkRendererBackend->gpuSceneDescriptorSet[vkRendererBackend->frameNumber % FRAME_OVERLAP];
}

void SK::VkRendererBackend::setViewport(State* vkRendererBackend, VkCommandBuffer cmd)
{
    VkViewport viewport = {};
    viewport.x = 0;
    viewport.y = 0;
    viewport.width = vkRendererBackend->drawExtent.width;
    viewport.height = vkRendererBackend->drawExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);
}

void SK::VkRendererBackend::setScissor(State* vkRendererBackend, VkCommandBuffer cmd)
{
    VkRect2D scissor = {};
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent.width = vkRendererBackend->drawExtent.width;
    scissor.extent.height = vkRendererBackend->drawExtent.height;
    vkCmdSetScissor(cmd, 0, 1, &scissor);
}

void SK::VkRendererBackend::createDrawAndDepthImages(State* vkRendererBackend)
{
    // draw image size will match the window
    VkExtent3D drawImageExtent = {
        vkRendererBackend->windowExtent.width,
        vkRendererBackend->windowExtent.height,
        1
    };

    // Initialize the draw image
    vkRendererBackend->drawImage = createImage(vkRendererBackend, drawImageExtent, VK_FORMAT_R16G16B16A16_SFLOAT, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    // Initialize the depth image
    vkRendererBackend->depthImage = createImage(vkRendererBackend, drawImageExtent, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

void SK::VkRendererBackend::destroyDrawAndDepthImages(State* vkRendererBackend)
{
    // Destroy the Draw Image
    vmaDestroyImage(vkRendererBackend->vmaAllocator, vkRendererBackend->drawImage.image, vkRendererBackend->drawImage.allocation);
    vkDestroyImageView(vkRendererBackend->device, vkRendererBackend->drawImage.imageView, nullptr);
    // Destroy the Depth Image
    vmaDestroyImage(vkRendererBackend->vmaAllocator, vkRendererBackend->depthImage.image, vkRendererBackend->depthImage.allocation);
    vkDestroyImageView(vkRendererBackend->device, vkRendererBackend->depthImage.imageView, nullptr);
}

SK::VkRendererBackend::FrameData& SK::VkRendererBackend::getCurrentFrameData(State* vkRendererBackend)
{
    return vkRendererBackend->frames[vkRendererBackend->frameNumber % FRAME_OVERLAP];
}

void SK::VkRendererBackend::registerOverlayPass(State* vkRendererBackend, OverlayPass pass)
{
    vkRendererBackend->overlayPasses.push_back(pass);
}

VkShaderModule SK::VkRendererBackend::getOrLoadShader(State* vkRendererBackend, const char* path)
{
    size_t hash = std::hash<std::string>{}(path);

    auto it = vkRendererBackend->shaderCache.find(hash);
    if(it != vkRendererBackend->shaderCache.end())
    {
        return it->second;
    }

    VkShaderModule shaderModule;
    if(!SK::VkUtil::loadShaderModule(vkRendererBackend->device, path, &shaderModule))
    {
        return VK_NULL_HANDLE;
    }

    vkRendererBackend->shaderCache[hash] = shaderModule;
    return shaderModule;
}
    
void SK::VkRendererBackend::clearShaderCache(State* vkRendererBackend)
{
    for(auto& [k, s] : vkRendererBackend->shaderCache)
    {
        vkDestroyShaderModule(vkRendererBackend->device, s, nullptr);
    }

    vkRendererBackend->shaderCache.clear();
}

VkPipelineLayout SK::VkRendererBackend::getOrCreatePipelineLayout(State* vkRendererBackend, const PipelineLayoutKey& key)
{
    size_t hash = hashPipelineLayoutKey(key);

    auto it = vkRendererBackend->pipelineLayoutCache.find(hash);
    if(it != vkRendererBackend->pipelineLayoutCache.end())
    {
        return it->second;
    }

    VkPipelineLayoutCreateInfo info = SK::VkInit::pipeline_layout_create_info();
    info.setLayoutCount = (uint32_t)key.setLayouts.size();
    info.pSetLayouts = key.setLayouts.data();
    info.pushConstantRangeCount = (uint32_t)key.pushConstantRanges.size();
    info.pPushConstantRanges = key.pushConstantRanges.data();

    VkPipelineLayout layout;
    VK_CHECK(vkCreatePipelineLayout(vkRendererBackend->device, &info, nullptr, &layout));

    vkRendererBackend->pipelineLayoutCache[hash] = layout;
    return layout;
}

void SK::VkRendererBackend::clearPipelineLayoutCache(State* vkRendererBackend)
{
    for(auto& [k, l] : vkRendererBackend->pipelineLayoutCache)
    {
        vkDestroyPipelineLayout(vkRendererBackend->device, l, nullptr);
    }

    vkRendererBackend->pipelineLayoutCache.clear();
}

VkPipeline SK::VkRendererBackend::getOrCreatePipeline(State* vkRendererBackend, const PipelineKey& key)
{
    size_t hash = hashPipelineKey(key);

    auto it = vkRendererBackend->pipelineCache.find(hash);
    if(it != vkRendererBackend->pipelineCache.end())
    {
        return it->second;
    }

    PipelineBuilder builder;
    builder.clear();

    builder.setShaders(
        vkRendererBackend->shaderCache[key.vertShader],
        vkRendererBackend->shaderCache[key.fragShader]
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

    VkPipeline pipeline = builder.buildPipeline(vkRendererBackend->device);

    vkRendererBackend->pipelineCache[hash] = pipeline;
    return pipeline;
}

void SK::VkRendererBackend::clearPipelineCache(State* vkRendererBackend)
{
    for(auto& [k, p] : vkRendererBackend->pipelineCache)
    {
        vkDestroyPipeline(vkRendererBackend->device, p, nullptr);
    }

    vkRendererBackend->pipelineCache.clear();
}

void SK::VkRendererBackend::m_initVulkan(State* vkRendererBackend)
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
    vkRendererBackend->instance = vkbInstance.instance;
    vkRendererBackend->debugMessenger = vkbInstance.debug_messenger;

    SDL_Vulkan_CreateSurface(vkRendererBackend->window, vkRendererBackend->instance, &vkRendererBackend->surface);

    // Vulkan 1.3 features
    VkPhysicalDeviceVulkan13Features features13{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    features13.dynamicRendering = true;
    features13.synchronization2 = true;

    // Vulkan 1.2 features
    VkPhysicalDeviceVulkan12Features features12{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    features12.bufferDeviceAddress = true;
    features12.descriptorIndexing = true;

    // Vulkan 1.0 features
    VkPhysicalDeviceFeatures features10{};
    features10.samplerAnisotropy = true;

    // Use vkbootstrap to select a gpu with Vulkan 1.3 and necessary features
    vkb::PhysicalDeviceSelector selector{vkbInstance};
    vkb::PhysicalDevice physicalDevice = selector
        .set_minimum_version(1, 3)
        .set_required_features_13(features13)
        .set_required_features_12(features12)
        .set_required_features(features10)
        .set_surface(vkRendererBackend->surface)
        .select()
        .value();

    // Create the final Vulkan device
    vkb::DeviceBuilder deviceBuilder{physicalDevice};
    vkb::Device vkbDevice = deviceBuilder.build().value();

    // Get the VKDevice handle used in the rest of the Vulkan application
    vkRendererBackend->device = vkbDevice.device;
    vkRendererBackend->chosenGPU = vkbDevice.physical_device;
    // Get the Graphics Queue
    vkRendererBackend->graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    vkRendererBackend->graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    // Initialize the memory allocator
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = vkRendererBackend->chosenGPU;
    allocatorInfo.device = vkRendererBackend->device;
    allocatorInfo.instance = vkRendererBackend->instance;
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&allocatorInfo, &vkRendererBackend->vmaAllocator);

    vkRendererBackend->mainDeletionQueue.pushFunction([=](){
        vmaDestroyAllocator(vkRendererBackend->vmaAllocator);
    });
}

void SK::VkRendererBackend::m_initSwapchain(State* vkRendererBackend)
{
    m_createSwapchain(vkRendererBackend, vkRendererBackend->windowExtent.width, vkRendererBackend->windowExtent.height);
}

void SK::VkRendererBackend::m_initCommands(State* vkRendererBackend)
{
    // Create the command pool and allow for resetting of individual command buffers
    VkCommandPoolCreateInfo commandPoolInfo = SK::VkInit::command_pool_create_info(vkRendererBackend->graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

    for(int i = 0; i < FRAME_OVERLAP; ++i)
    {
        VK_CHECK(vkCreateCommandPool(vkRendererBackend->device, &commandPoolInfo, nullptr, &vkRendererBackend->frames[i].commandPool));
        // Allocate the default command buffer that will be used for rendering
        VkCommandBufferAllocateInfo cmdAllocInfo = SK::VkInit::command_buffer_allocate_info(vkRendererBackend->frames[i].commandPool, 1);
        VK_CHECK(vkAllocateCommandBuffers(vkRendererBackend->device, &cmdAllocInfo, &vkRendererBackend->frames[i].mainCommandBuffer));
    }

    // Immediate commands
    VK_CHECK(vkCreateCommandPool(vkRendererBackend->device, &commandPoolInfo, nullptr, &vkRendererBackend->immediateCommandPool));

    // Allocate a command buffer for immediate submits
    VkCommandBufferAllocateInfo cmdAllocInfo = SK::VkInit::command_buffer_allocate_info(vkRendererBackend->immediateCommandPool, 1);

    VK_CHECK(vkAllocateCommandBuffers(vkRendererBackend->device, &cmdAllocInfo, &vkRendererBackend->immediateCommandBuffer));

    vkRendererBackend->mainDeletionQueue.pushFunction([=](){
        vkDestroyCommandPool(vkRendererBackend->device, vkRendererBackend->immediateCommandPool, nullptr);
    });
}

void SK::VkRendererBackend::m_initSyncStructures(State* vkRendererBackend)
{
    //create syncronization structures
    //one fence to control when the gpu has finished rendering the frame,
    //and 2 semaphores to syncronize rendering with swapchain
    //we want the fence to start signalled so we can wait on it on the first frame
    VkFenceCreateInfo fenceCreateInfo = SK::VkInit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VkSemaphoreCreateInfo semaphoreCreateInfo = SK::VkInit::semaphore_create_info();

    for(int i = 0; i < FRAME_OVERLAP; ++i)
    {
        VK_CHECK(vkCreateFence(vkRendererBackend->device, &fenceCreateInfo, nullptr, &vkRendererBackend->frames[i].renderFence));

        VK_CHECK(vkCreateSemaphore(vkRendererBackend->device, &semaphoreCreateInfo, nullptr, &vkRendererBackend->frames[i].swapchainSemaphore));
        VK_CHECK(vkCreateSemaphore(vkRendererBackend->device, &semaphoreCreateInfo, nullptr, &vkRendererBackend->frames[i].renderSemaphore));
    }

    // Fence for the immediate command buffers
    VK_CHECK(vkCreateFence(vkRendererBackend->device, &fenceCreateInfo, nullptr, &vkRendererBackend->immeadiateFence));
    vkRendererBackend->mainDeletionQueue.pushFunction([=](){
        vkDestroyFence(vkRendererBackend->device, vkRendererBackend->immeadiateFence, nullptr);
    });
}

void SK::VkRendererBackend::m_createSwapchain(State* vkRendererBackend, uint32_t width, uint32_t height)
{
    vkb::SwapchainBuilder swapchainBuilder{ vkRendererBackend->chosenGPU, vkRendererBackend->device, vkRendererBackend->surface};

    vkRendererBackend->swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;

    vkb::Swapchain vkbSwapchain = swapchainBuilder
        .set_desired_format(VkSurfaceFormatKHR{.format = vkRendererBackend->swapchainImageFormat, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(width, height)
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .build()
        .value();

    
    vkRendererBackend->swapchainExtent = vkbSwapchain.extent;
    // Store the swapchain and its related images
    vkRendererBackend->swapchain = vkbSwapchain.swapchain;
    vkRendererBackend->swapchainImages = vkbSwapchain.get_images().value();
    vkRendererBackend->swapchainImageViews = vkbSwapchain.get_image_views().value();
}

void SK::VkRendererBackend::m_destroySwapchain(State* vkRendererBackend)
{
    // Deleting the swapchain deletes the images it holds internally.
    vkDestroySwapchainKHR(vkRendererBackend->device, vkRendererBackend->swapchain, nullptr);

    // Destroy the swapchain resources
    for(int i = 0; i < vkRendererBackend->swapchainImageViews.size(); ++i)
    {
        vkDestroyImageView(vkRendererBackend->device, vkRendererBackend->swapchainImageViews[i], nullptr);
    }

    vkRendererBackend->swapchainImages.clear();
    vkRendererBackend->swapchainImageViews.clear();
}

void SK::VkRendererBackend::handleWindowResize(State* vkRendererBackend)
{
    // Don't change the images and views while the gpu is still handling them
    vkDeviceWaitIdle(vkRendererBackend->device);

    int w, h;
    SDL_GetWindowSize(vkRendererBackend->window, &w, &h);
    vkRendererBackend->windowExtent.width = w;
    vkRendererBackend->windowExtent.height = h;

    // Recreate swapchain and draw, depth images.
    m_destroySwapchain(vkRendererBackend);
    m_createSwapchain(vkRendererBackend, vkRendererBackend->windowExtent.width, vkRendererBackend->windowExtent.height);
    destroyDrawAndDepthImages(vkRendererBackend);
    createDrawAndDepthImages(vkRendererBackend);

    vkRendererBackend->windowResizeRequested = false;
}

void SK::VkRendererBackend::m_initDescriptors(State* vkRendererBackend)
{
    // Create the global growable descriptor allocator 
    std::vector<DescriptorAllocatorGrowable::PoolSize> sizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 }
    };

    vkRendererBackend->globalDescriptorAllocator.init(vkRendererBackend->device, 10, sizes);
    
    // The descriptor set layout for the main draw image
    {
        DescriptorLayoutBuilder builder;
        builder.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        vkRendererBackend->drawImageDescriptorSetLayout = builder.build(vkRendererBackend->device, VK_SHADER_STAGE_COMPUTE_BIT);
    }

    // The descriptor set layout for single texture display
    {
        DescriptorLayoutBuilder builder;
        builder.addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        vkRendererBackend->displayTextureDescriptorSetLayout = builder.build(vkRendererBackend->device, VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    // Descriptor set layout for the scene data
    {
        DescriptorLayoutBuilder builder;
        builder.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        vkRendererBackend->gpuSceneDataDescriptorLayout = builder.build(vkRendererBackend->device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    // Allocate a descriptor set for the draw image
    vkRendererBackend->drawImageDescriptorSet = vkRendererBackend->globalDescriptorAllocator.allocate(vkRendererBackend->device, vkRendererBackend->drawImageDescriptorSetLayout);

    {
        DescriptorWriter writer;
        writer.writeImage(0, vkRendererBackend->drawImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        writer.updateSet(vkRendererBackend->device, vkRendererBackend->drawImageDescriptorSet);
    }

    // Add the descriptor allocator and layout destructors to the deletion queue
    vkRendererBackend->mainDeletionQueue.pushFunction([=](){
        vkRendererBackend->globalDescriptorAllocator.destroyPools(vkRendererBackend->device);

        vkDestroyDescriptorSetLayout(vkRendererBackend->device, vkRendererBackend->drawImageDescriptorSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(vkRendererBackend->device, vkRendererBackend->displayTextureDescriptorSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(vkRendererBackend->device, vkRendererBackend->gpuSceneDataDescriptorLayout, nullptr);
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

        vkRendererBackend->frames[i].frameDescriptorAllocator = DescriptorAllocatorGrowable{};
        vkRendererBackend->frames[i].frameDescriptorAllocator.init(vkRendererBackend->device, 1000, framePoolSizes);

        // Pools in the frame descriptor allocators must be destroyed with the vkRendererBackend shutdown (not with frame shutdown)
        vkRendererBackend->mainDeletionQueue.pushFunction([=]() {
            vkRendererBackend->frames[i].frameDescriptorAllocator.destroyPools(vkRendererBackend->device);
        });
    }
}

void SK::VkRendererBackend::m_initMaterialLayouts(State* vkRendererBackend)
{
    GLTFMetallicRoughnessMaterial::BuildMaterialLayout(vkRendererBackend);
}

void SK::VkRendererBackend::m_clearMaterialLayouts(State* vkRendererBackend)
{
    GLTFMetallicRoughnessMaterial::ClearMaterialLayout(vkRendererBackend->device);
}

void SK::VkRendererBackend::m_initDefaultData(State* vkRendererBackend)
{
    // Default textures
    // 3 default textures 1 pixel each
    uint32_t white = glm::packUnorm4x8(glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
    vkRendererBackend->whiteImage = createImage(vkRendererBackend, (void*)&white, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t grey = glm::packUnorm4x8(glm::vec4(0.66f, 0.66f, 0.66f, 1.0f));
    vkRendererBackend->greyImage = createImage(vkRendererBackend, (void*)&grey, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t black = glm::packUnorm4x8(glm::vec4(0.0f, 0.0f, 0.0f, 0.0f));
    vkRendererBackend->blackImage =createImage(vkRendererBackend, (void*)&black, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

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

    vkRendererBackend->errorCheckerboardImage = createImage(vkRendererBackend, pixels.data(), VkExtent3D{ 16, 16, 1 }, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    // Default samplers
    VkSamplerCreateInfo samplerInfo = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };

    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    vkCreateSampler(vkRendererBackend->device, &samplerInfo, nullptr, &vkRendererBackend->defaultSamplerNearest);

    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    vkCreateSampler(vkRendererBackend->device, &samplerInfo, nullptr, &vkRendererBackend->defaultSamplerLinear);

    vkRendererBackend->mainDeletionQueue.pushFunction([=]() {
        destroyImage(vkRendererBackend, vkRendererBackend->whiteImage);
        destroyImage(vkRendererBackend, vkRendererBackend->greyImage);
        destroyImage(vkRendererBackend, vkRendererBackend->blackImage);
        destroyImage(vkRendererBackend, vkRendererBackend->errorCheckerboardImage);

        vkDestroySampler(vkRendererBackend->device, vkRendererBackend->defaultSamplerNearest, nullptr);
        vkDestroySampler(vkRendererBackend->device, vkRendererBackend->defaultSamplerLinear, nullptr);
    });

    // Default material data
    GLTFMetallicRoughnessMaterial::MaterialResources defaultMaterialResources;
    defaultMaterialResources.colorImage = vkRendererBackend->whiteImage;
    defaultMaterialResources.colorSampler = vkRendererBackend->defaultSamplerLinear;
    defaultMaterialResources.metalRoughnessImage = vkRendererBackend->whiteImage;
    defaultMaterialResources.metalRoughnessSampler = vkRendererBackend->defaultSamplerLinear;
    
    AllocatedBuffer materialConstantsBuffer = createBuffer(vkRendererBackend, sizeof(GLTFMetallicRoughnessMaterial::MaterialConstants), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    // Write the buffer
    GLTFMetallicRoughnessMaterial::MaterialConstants* pMaterialConstantsBuffer = static_cast<GLTFMetallicRoughnessMaterial::MaterialConstants*>(materialConstantsBuffer.allocation->GetMappedData());
    pMaterialConstantsBuffer->colorFactors = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    pMaterialConstantsBuffer->metalRoughnessFactors = glm::vec4(1.0f, 0.5f, 0.0f, 0.0f);

    vkRendererBackend->mainDeletionQueue.pushFunction([=]() {
        destroyBuffer(vkRendererBackend, materialConstantsBuffer);
    });

    defaultMaterialResources.dataBuffer = materialConstantsBuffer.buffer;
    defaultMaterialResources.dataBufferOffset = 0;

    vkRendererBackend->defaultMaterialInstance = GLTFMetallicRoughnessMaterial::CreateInstance(vkRendererBackend->device, MaterialPass::Opaque, defaultMaterialResources, vkRendererBackend->globalDescriptorAllocator);
}

void SK::VkRendererBackend::m_initGlobalSceneBuffer(State* vkRendererBackend)
{
    for(int i = 0; i < FRAME_OVERLAP; ++i)
    {
        // Allocate a new uniform buffer for scene data (allocating on VRAM that CPU can write to directly. It is limited but it is perfect for allocating reasonable amounts that are dynamic)
        vkRendererBackend->gpuSceneDataBuffer[i] = createBuffer(vkRendererBackend, sizeof(GPUSceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        vkRendererBackend->mainDeletionQueue.pushFunction([=]() {
            destroyBuffer(vkRendererBackend, vkRendererBackend->gpuSceneDataBuffer[i]);
        });

        // Create a descriptor set for the uniform data
        vkRendererBackend->gpuSceneDescriptorSet[i] = vkRendererBackend->globalDescriptorAllocator.allocate(vkRendererBackend->device, vkRendererBackend->gpuSceneDataDescriptorLayout);
        DescriptorWriter writer;
        writer.writeBuffer(0, vkRendererBackend->gpuSceneDataBuffer[i].buffer, sizeof(GPUSceneData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        writer.updateSet(vkRendererBackend->device, vkRendererBackend->gpuSceneDescriptorSet[i]);
    }
}
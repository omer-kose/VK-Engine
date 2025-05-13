#include "vk_engine.h"

#include <chrono>
#include <thread>

#include <SDL.h>
#include <SDL_vulkan.h>

#include <Core/vk_initializers.h>
#include <Core/vk_types.h>
#include <Core/vk_images.h>
#include <Core/vk_pipelines.h>
#include "VkBootstrap.h"

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

#include <glm/gtx/transform.hpp>


constexpr bool bUseValidationLayers = true;

VulkanEngine* loadedEngine = nullptr;

VulkanEngine& VulkanEngine::Get() { return *loadedEngine; }
void VulkanEngine::init()
{
    // only one engine initialization is allowed with the application.
    assert(loadedEngine == nullptr);
    loadedEngine = this;

    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

    window = SDL_CreateWindow(
        "Vulkan Engine",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        windowExtent.width,
        windowExtent.height,
        window_flags
    );

    // Vulkan Bootstrapping
    m_initVulkan();
    m_initSwapchain();
    m_initCommands();
    m_initSyncStructures();

    m_initDescriptors();

    m_initMaterialLayouts();

    m_initPasses();

    m_initImgui();

    m_initDefaultData();

    m_initGlobalSceneBuffer();

    // everything went fine
    isInitialized = true;

    m_initCamera(glm::vec3(30.f, -0.0f, -85.0f), 0.0f, 0.0f);
    
    m_loadSceneData();
}

void VulkanEngine::cleanup()
{
    if(isInitialized) 
    {
        // make sure that GPU is done with the command buffers
        vkDeviceWaitIdle(device);

        for(int i = 0; i < FRAME_OVERLAP; ++i)
        {
            // Destroy sync objects
            vkDestroyFence(device, frames[i].renderFence, nullptr);
            vkDestroySemaphore(device, frames[i].swapchainSemaphore, nullptr);
            vkDestroySemaphore(device, frames[i].renderSemaphore, nullptr);

            // It’s not possible to individually destroy VkCommandBuffer, destroying their parent pool will destroy all of the command buffers allocated from it.
            vkDestroyCommandPool(device, frames[i].commandPool, nullptr);

            frames[i].deletionQueue.flush();
        }

        loadedScenes.clear();

        m_clearMaterialLayouts();

        m_clearPassResources();

        mainDeletionQueue.flush();

        m_destroySwapchain();

        vkDestroyDevice(device, nullptr);
        vkDestroySurfaceKHR(instance, surface, nullptr);

        vkb::destroy_debug_utils_messenger(instance, debugMessenger);
        vkDestroyInstance(instance, nullptr);
        SDL_DestroyWindow(window);
    }

    // clear engine pointer
    loadedEngine = nullptr;
}

void VulkanEngine::draw()
{
    FrameData& currentFrame = getCurrentFrame();
    // Wait until the GPU has finished rendering the last frame of the same modularity (0->1->2->3  wait on 2 for 0 and wait on 3 for 1 and so on)
    VK_CHECK(vkWaitForFences(device, 1, &currentFrame.renderFence, true, 1000000000));

    currentFrame.deletionQueue.flush();
    currentFrame.frameDescriptorAllocator.clearPools(device);

    // To be able to use the same fence it must be reset after use
    VK_CHECK(vkResetFences(device, 1, &currentFrame.renderFence));
    
    // Request an available image from the swapchain. swapchainSemaphore is signaled once it has finished presenting the image so it can be used again.
    // More detailed description of how vkAcquireNextImageKHR works: https://stackoverflow.com/questions/60419749/why-does-vkacquirenextimagekhr-never-block-my-thread
    uint32_t swapchainImageIndex;
    VkResult acquireResult = vkAcquireNextImageKHR(device, swapchain, 1000000000, currentFrame.swapchainSemaphore, nullptr, &swapchainImageIndex);
    if(acquireResult == VK_ERROR_OUT_OF_DATE_KHR)
    {
        resizeRequested = true;
        return;
    }

    // Extent of the image that we are going to draw onto
    drawExtent.width = std::min(drawImage.imageExtent.width, swapchainExtent.width) * renderScale;
    drawExtent.height = std::min(drawImage.imageExtent.height, swapchainExtent.height) * renderScale;
    
    // Vulkan handles are just a 64 bit handles/pointers, so its fine to copy them around, but remember that their actual data is handled by vulkan itself.
    VkCommandBuffer cmd = currentFrame.mainCommandBuffer;

    // Now we are sure that command is executed, we can safely reset it and begin recording again
    VK_CHECK(vkResetCommandBuffer(cmd, 0));

    // Begin the command buffer recording. We will submit this command buffer exactly once, so we let Vulkan know that
    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    // Start the command buffer recording
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    // Transition depth image to optimal depth layout
    vkutil::transitionImage(cmd, depthImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    
    // encode drawing commands except ImGui
    drawMain(cmd);

    // Transition the draw image and the swapchain image into their correct layouts
    vkutil::transitionImage(cmd, drawImage.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkutil::transitionImage(cmd, swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // Execute a copy operation from the draw image into the swapchain image
    vkutil::copyImageToImage(cmd, drawImage.image, swapchainImages[swapchainImageIndex], drawExtent, swapchainExtent);

    // After drawing, we need to draw ImGui on top of the swapchain image, so transition the swapchain image into optimal drawing layout
    vkutil::transitionImage(cmd, swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    drawImgui(cmd, swapchainImageViews[swapchainImageIndex]);

    // Transition swapchain image into the presentation layout
    vkutil::transitionImage(cmd, swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

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
    VK_CHECK(vkQueueSubmit2(graphicsQueue, 1, &submit, currentFrame.renderFence));

    // Prepare the presentation
    // We will wait on the renderSemaphore so that it will be guaranteed that the rendering has been finished and the swapchain image is ready to be presented
    VkPresentInfoKHR presentInfo = {.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, .pNext = nullptr};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &currentFrame.renderSemaphore;
    presentInfo.pImageIndices = &swapchainImageIndex;

    VkResult presentResult = vkQueuePresentKHR(graphicsQueue, &presentInfo);
    if(presentResult == VK_ERROR_OUT_OF_DATE_KHR)
    {
        resizeRequested = true;
    }

    // Increase the number of frames drawn
    ++frameNumber;
}

void VulkanEngine::drawMain(VkCommandBuffer cmd)
{
    updateSceneBuffer();

    // When rendering geometry we need to use COLOR_ATTACHMENT_OPTIMAL as it is the most optimal layout for rendering with graphics pipeline
    vkutil::transitionImage(cmd, drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // Begin a renderpass connected to the draw image
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(drawImage.imageView, &colorAttachmentClearValue, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo depthAttachment = vkinit::depth_attachment_info(depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    VkRenderingInfo renderInfo = vkinit::rendering_info(drawExtent, &colorAttachment, &depthAttachment);
    vkCmdBeginRendering(cmd, &renderInfo);

    auto start = std::chrono::system_clock::now();

    drawGeometry(cmd);

    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    stats.geometryDrawRecordTime = elapsed.count() / 1000.f;

    vkCmdEndRendering(cmd);
}

void VulkanEngine::drawGeometry(VkCommandBuffer cmd)
{
    // Go through all the graphics passes and execute them
    GLTFMetallicPass::Execute(this, cmd);

    // Drawing is done context can be cleared
    mainDrawContext.opaqueGLTFSurfaces.clear();
    mainDrawContext.transparentGLTFSurfaces.clear();
}

void VulkanEngine::drawImgui(VkCommandBuffer cmd, VkImageView targetImageView)
{
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(targetImageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingInfo renderInfo = vkinit::rendering_info(swapchainExtent, &colorAttachment, nullptr);

    vkCmdBeginRendering(cmd, &renderInfo);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRendering(cmd);
}

void VulkanEngine::updateScene()
{
    auto start = std::chrono::system_clock::now();

    mainCamera.update();

    loadedScenes["structure"]->registerDraw(glm::mat4(1.0f), mainDrawContext);

    sceneData.view = mainCamera.getViewMatrix();
    // camera projection
    sceneData.proj = glm::perspectiveRH_ZO(glm::radians(70.f), (float)windowExtent.width / (float)windowExtent.height, 0.1f, 10000.f);

    // invert the Y direction on projection matrix so that we are more similar
    // to opengl and gltf axis
    sceneData.proj[1][1] *= -1;
    sceneData.viewproj = sceneData.proj * sceneData.view;

    //some default lighting parameters
    sceneData.ambientColor = glm::vec4(0.1f);
    sceneData.sunlightColor = glm::vec4(1.0f);
    sceneData.sunlightDirection = glm::vec4(0.0f, 1.0f, 0.5f, 1.0f);

    auto end = std::chrono::system_clock::now();
    // Convert to microseconds (integer), then come back to miliseconds
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    stats.sceneUpdateTime = elapsed.count() / 1000.0f;
}

void VulkanEngine::run()
{
    SDL_Event e;
    bool bQuit = false;

    // main loop
    while(!bQuit) 
    {
        // Begin frame time clock
        auto start = std::chrono::system_clock::now();

        // Handle events on queue
        while(SDL_PollEvent(&e) != 0) 
        {
            // close the window when user alt-f4s or clicks the X button
            if(e.type == SDL_QUIT)
                bQuit = true;

            if(e.type == SDL_WINDOWEVENT) 
            {
                if(e.window.event == SDL_WINDOWEVENT_MINIMIZED) 
                {
                    freezeRendering = true;
                }
                if(e.window.event == SDL_WINDOWEVENT_RESTORED) 
                {
                    freezeRendering = false;
                }
            }

            mainCamera.processSDLEvent(e);
            // send SDL event to ImGui for processing
            ImGui_ImplSDL2_ProcessEvent(&e);
        }

        // do not draw if we are minimized
        if(freezeRendering) 
        {
            // throttle the speed to avoid the endless spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        if(resizeRequested)
        {
            m_resizeSwapchain();
        }

        // ImGui new frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Stats");

        ImGui::Text("frametime %f ms", stats.frameTime);
        ImGui::Text("geometry draw recording time %f ms", stats.geometryDrawRecordTime);
        ImGui::Text("update time %f ms", stats.sceneUpdateTime);
        ImGui::Text("triangles %i", stats.triangleCount);
        ImGui::Text("draws %i", stats.drawCallCount);
        ImGui::End();

        // Make ImGui calculate internal draw structures
        ImGui::Render();

        updateScene();

        draw();

        auto end = std::chrono::system_clock::now();
        // Convert to microseconds (integer), then come back to miliseconds
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        stats.frameTime = elapsed.count() / 1000.0f;
    }
}

void VulkanEngine::immediateSubmit(std::function<void(VkCommandBuffer cmd)>&& function)
{
    // Before starting submitting and waiting on the fence reset them
    VK_CHECK(vkResetFences(device, 1, &immeadiateFence));
    VK_CHECK(vkResetCommandBuffer(immediateCommandBuffer, 0));
    // Prepare the immediate command buffer for executing function given as the param
    VkCommandBuffer cmd = immediateCommandBuffer;
    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));
    function(cmd);
    VK_CHECK(vkEndCommandBuffer(cmd));

    // Submit
    VkCommandBufferSubmitInfo cmdSubmitInfo = vkinit::command_buffer_submit_info(cmd);
    VkSubmitInfo2 submitInfo = vkinit::submit_info(&cmdSubmitInfo, nullptr, nullptr);
    VK_CHECK(vkQueueSubmit2(graphicsQueue, 1, &submitInfo, immeadiateFence));

    // Wait on the fence until the command buffer finished executing
    VK_CHECK(vkWaitForFences(device, 1, &immeadiateFence, true, 9999999999));
}

AllocatedBuffer VulkanEngine::createBuffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage)
{
    // Allocate buffer
    VkBufferCreateInfo bufferInfo = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = nullptr };
    bufferInfo.size = allocSize;
    bufferInfo.usage = usage;

    VmaAllocationCreateInfo vmaAllocInfo = {};
    vmaAllocInfo.usage = memoryUsage;
    vmaAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    AllocatedBuffer newBuffer;

    VK_CHECK(vmaCreateBuffer(vmaAllocator, &bufferInfo, &vmaAllocInfo, &newBuffer.buffer, &newBuffer.allocation, &newBuffer.allocInfo));

    return newBuffer;
}

void VulkanEngine::destroyBuffer(const AllocatedBuffer& buffer)
{
    vmaDestroyBuffer(vmaAllocator, buffer.buffer, buffer.allocation);
}

AllocatedImage VulkanEngine::createImage(VkExtent3D imageExtent, VkFormat format, VkImageUsageFlags usage, bool mipMapped)
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
    VK_CHECK(vmaCreateImage(vmaAllocator, &imgInfo, &allocInfo, &newImage.image, &newImage.allocation, nullptr));

    // Defaulting to the color aspect unless depth format is given
    VkImageAspectFlags aspectFlag = VK_IMAGE_ASPECT_COLOR_BIT; 
    if(format == VK_FORMAT_D32_SFLOAT) // if the format is the depth format
    {
        aspectFlag = VK_IMAGE_ASPECT_DEPTH_BIT;
    }

    // Create the image-view for the image
    VkImageViewCreateInfo viewInfo = vkinit::imageview_create_info(format, newImage.image, aspectFlag);
    viewInfo.subresourceRange.levelCount = imgInfo.mipLevels;

    VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &newImage.imageView));

    return newImage;
}

AllocatedImage VulkanEngine::createImage(void* data, VkExtent3D imageExtent, VkFormat format, VkImageUsageFlags usage, bool mipMapped)
{
    // Hardcoding the textures to be RGBA 8 bit format. This should be sufficient as most of the textures are in that format.
    size_t dataSize = imageExtent.depth * imageExtent.width * imageExtent.height * 4;
    AllocatedBuffer uploadBuffer = createBuffer(dataSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    memcpy(uploadBuffer.allocInfo.pMappedData, data, dataSize);

    // aside from the original usage also allow copying data into and from it.
    AllocatedImage newImage = createImage(imageExtent, format, usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, mipMapped);

    // Perform a buffer to image copy.
    immediateSubmit([&](VkCommandBuffer cmd) {
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

    destroyBuffer(uploadBuffer);

    return newImage;
}

void VulkanEngine::destroyImage(const AllocatedImage& img)
{
    vkDestroyImageView(device, img.imageView, nullptr);
    vmaDestroyImage(vmaAllocator, img.image, img.allocation);
}

GPUMeshBuffers VulkanEngine::uploadMesh(std::span<Vertex> vertices, std::span<uint32_t> indices)
{
    const size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
    const size_t indexBufferSize = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers meshBuffers;

    // Create the vertex buffer and fetch the device address of it
    meshBuffers.vertexBuffer = createBuffer(vertexBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    VkBufferDeviceAddressInfo deviceAddressInfo{ .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = meshBuffers.vertexBuffer.buffer };
    meshBuffers.vertexBufferAddress = vkGetBufferDeviceAddress(device, &deviceAddressInfo);

    // Create the index buffer
    meshBuffers.indexBuffer = createBuffer(indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    AllocatedBuffer staging = createBuffer(vertexBufferSize + indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    void* data = staging.allocation->GetMappedData();

    // Copy Vertex Buffer
    memcpy(data, vertices.data(), vertexBufferSize);
    // Copy Index Buffer
    memcpy((char*)data + vertexBufferSize, indices.data(), indexBufferSize);

    immediateSubmit([&](VkCommandBuffer cmd) {
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

    destroyBuffer(staging);

    return meshBuffers;
}

/*
    Both update and bind scene buffer functions must be called after the frame fence waits as it will be guaranteed that the frame is done being used by GPU. Otherwise, the data can be corrupted. 
    (calling in drawMain() will suffice)
*/
void VulkanEngine::updateSceneBuffer()
{
    // Update the scene buffer
    GPUSceneData* pGpuSceneDataBuffer = (GPUSceneData*)gpuSceneDataBuffer[frameNumber % FRAME_OVERLAP].allocation->GetMappedData();
    *pGpuSceneDataBuffer = sceneData;
}

VkDescriptorSetLayout VulkanEngine::getSceneDescriptorLayout() const
{
    return sceneDataDescriptorLayout;
}

VkDescriptorSet VulkanEngine::getSceneBufferDescriptorSet() const
{
    return sceneDescriptorSet[frameNumber % FRAME_OVERLAP];
}

void VulkanEngine::setViewport(VkCommandBuffer cmd)
{
    VkViewport viewport = {};
    viewport.x = 0;
    viewport.y = 0;
    viewport.width = drawExtent.width;
    viewport.height = drawExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);
}

void VulkanEngine::setScissor(VkCommandBuffer cmd)
{
    VkRect2D scissor = {};
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent.width = drawExtent.width;
    scissor.extent.height = drawExtent.height;
    vkCmdSetScissor(cmd, 0, 1, &scissor);
}

const DrawContext* VulkanEngine::getDrawContext() const
{
    return &mainDrawContext;
}

void VulkanEngine::m_initVulkan()
{
    vkb::InstanceBuilder builder;

    // Create the Vulkan instance with basic debug features.
    auto instRet = builder.set_app_name("Vulkan Engine")
        .request_validation_layers(bUseValidationLayers)
        .use_default_debug_messenger()
        .require_api_version(1, 3, 0)
        .build();

    vkb::Instance vkbInstance = instRet.value();

    // Grab the instance
    instance = vkbInstance.instance;
    debugMessenger = vkbInstance.debug_messenger;

    SDL_Vulkan_CreateSurface(window, instance, &surface);

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
        .set_surface(surface)
        .select()
        .value();

    // Create the final Vulkan device
    vkb::DeviceBuilder deviceBuilder{physicalDevice};
    vkb::Device vkbDevice = deviceBuilder.build().value();

    // Get the VKDevice handle used in the rest of the Vulkan application
    device = vkbDevice.device;
    chosenGPU = vkbDevice.physical_device;
    // Get the Graphics Queue
    graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    // Initialize the memory allocator
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = chosenGPU;
    allocatorInfo.device = device;
    allocatorInfo.instance = instance;
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&allocatorInfo, &vmaAllocator);

    mainDeletionQueue.pushFunction([=](){
        vmaDestroyAllocator(vmaAllocator);
    });
}

void VulkanEngine::m_initSwapchain()
{
    m_createSwapchain(windowExtent.width, windowExtent.height);

    // draw image size will match the window
    VkExtent3D drawImageExtent = {
        windowExtent.width,
        windowExtent.height,
        1
    };

    drawImage.imageExtent = drawImageExtent;

    // Hardcoding the draw format to 16 bit float
    drawImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

    VkImageUsageFlags drawImageUsageFlags{};
    drawImageUsageFlags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    drawImageUsageFlags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    drawImageUsageFlags |= VK_IMAGE_USAGE_STORAGE_BIT;
    drawImageUsageFlags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    VkImageCreateInfo drawImageInfo = vkinit::image_create_info(drawImage.imageFormat, drawImageUsageFlags, drawImage.imageExtent);

    // For the draw image, we want to allocate it from the gpu local memory
    VmaAllocationCreateInfo imageAllocInfo = {};
    imageAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    imageAllocInfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Allocate and create the image
    vmaCreateImage(vmaAllocator, &drawImageInfo, &imageAllocInfo, &drawImage.image, &drawImage.allocation, nullptr);

    // Build an image-view for the draw image to use for rendering
    VkImageViewCreateInfo drawImageViewInfo = vkinit::imageview_create_info(drawImage.imageFormat, drawImage.image, VK_IMAGE_ASPECT_COLOR_BIT);

    VK_CHECK(vkCreateImageView(device, &drawImageViewInfo, nullptr, &drawImage.imageView));

    // Initialize the depth image
    depthImage.imageFormat = VK_FORMAT_D32_SFLOAT; // one-component, 32-bit signed floating-point format that has 32 bits in the depth component
    depthImage.imageExtent = drawImageExtent;
    VkImageUsageFlags depthImageUsages{};
    depthImageUsages |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    VkImageCreateInfo depthImageInfo = vkinit::image_create_info(depthImage.imageFormat, depthImageUsages, depthImage.imageExtent);
    vmaCreateImage(vmaAllocator, &depthImageInfo, &imageAllocInfo, &depthImage.image, &depthImage.allocation, nullptr);
    VkImageViewCreateInfo depthViewInfo = vkinit::imageview_create_info(depthImage.imageFormat, depthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT);
    VK_CHECK(vkCreateImageView(device, &depthViewInfo, nullptr, &depthImage.imageView));

    // Add the resources to the deletion queue
    mainDeletionQueue.pushFunction([=](){
        // Destroy the Draw Image
        vmaDestroyImage(vmaAllocator, drawImage.image, drawImage.allocation);
        vkDestroyImageView(device, drawImage.imageView, nullptr);
        // Destroy the Depth Image
        vmaDestroyImage(vmaAllocator, depthImage.image, depthImage.allocation);
        vkDestroyImageView(device, depthImage.imageView, nullptr);
    });
}

void VulkanEngine::m_initCommands()
{
    // Create the command pool and allow for resetting of individual command buffers
    VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

    for(int i = 0; i < FRAME_OVERLAP; ++i)
    {
        VK_CHECK(vkCreateCommandPool(device, &commandPoolInfo, nullptr, &frames[i].commandPool));
        // Allocate the default command buffer that will be used for rendering
        VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(frames[i].commandPool, 1);
        VK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &frames[i].mainCommandBuffer));
    }

    // Immediate commands
    VK_CHECK(vkCreateCommandPool(device, &commandPoolInfo, nullptr, &immediateCommandPool));

    // Allocate a command buffer for immediate submits
    VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(immediateCommandPool, 1);

    VK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &immediateCommandBuffer));

    mainDeletionQueue.pushFunction([=](){
        vkDestroyCommandPool(device, immediateCommandPool, nullptr);
    });
}

void VulkanEngine::m_initSyncStructures()
{
    //create syncronization structures
    //one fence to control when the gpu has finished rendering the frame,
    //and 2 semaphores to syncronize rendering with swapchain
    //we want the fence to start signalled so we can wait on it on the first frame
    VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

    for(int i = 0; i < FRAME_OVERLAP; ++i)
    {
        VK_CHECK(vkCreateFence(device, &fenceCreateInfo, nullptr, &frames[i].renderFence));

        VK_CHECK(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &frames[i].swapchainSemaphore));
        VK_CHECK(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &frames[i].renderSemaphore));
    }

    // Fence for the immediate command buffers
    VK_CHECK(vkCreateFence(device, &fenceCreateInfo, nullptr, &immeadiateFence));
    mainDeletionQueue.pushFunction([=](){
        vkDestroyFence(device, immeadiateFence, nullptr);
    });
}

void VulkanEngine::m_createSwapchain(uint32_t width, uint32_t height)
{
    vkb::SwapchainBuilder swapchainBuilder{chosenGPU, device, surface};

    swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;

    vkb::Swapchain vkbSwapchain = swapchainBuilder
        .set_desired_format(VkSurfaceFormatKHR{.format = swapchainImageFormat, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(width, height)
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .build()
        .value();

    
    swapchainExtent = vkbSwapchain.extent;
    // Store the swapchain and its related images
    swapchain = vkbSwapchain.swapchain;
    swapchainImages = vkbSwapchain.get_images().value();
    swapchainImageViews = vkbSwapchain.get_image_views().value();
}

void VulkanEngine::m_destroySwapchain()
{
    // Deleting the swapchain deletes the images it holds internally.
    vkDestroySwapchainKHR(device, swapchain, nullptr);

    // Destroy the swapchain resources
    for(int i = 0; i < swapchainImageViews.size(); ++i)
    {
        vkDestroyImageView(device, swapchainImageViews[i], nullptr);
    }

    swapchainImages.clear();
    swapchainImageViews.clear();
}

void VulkanEngine::m_resizeSwapchain()
{
    // Don't change the images and views while the gpu is still handling them
    vkDeviceWaitIdle(device);

    m_destroySwapchain();

    int w, h;
    SDL_GetWindowSize(window, &w, &h);
    windowExtent.width = w;
    windowExtent.height = h;

    m_createSwapchain(windowExtent.width, windowExtent.height);

    resizeRequested = false;
}

void VulkanEngine::m_initDescriptors()
{
    // Create the global growable descriptor allocator 
    std::vector<DescriptorAllocatorGrowable::PoolSize> sizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 }
    };

    globalDescriptorAllocator.init(device, 10, sizes);
    
    // The descriptor set layout for the main draw image
    {
        DescriptorLayoutBuilder builder;
        builder.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        drawImageDescriptorSetLayout = builder.build(device, VK_SHADER_STAGE_COMPUTE_BIT);
    }

    // The descriptor set layout for single texture display
    {
        DescriptorLayoutBuilder builder;
        builder.addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        displayTextureDescriptorSetLayout = builder.build(device, VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    // Descriptor set layout for the scene data
    {
        DescriptorLayoutBuilder builder;
        builder.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        sceneDataDescriptorLayout = builder.build(device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    // Allocate a descriptor set for the draw image
    drawImageDescriptorSet = globalDescriptorAllocator.allocate(device, drawImageDescriptorSetLayout);

    {
        DescriptorWriter writer;
        writer.writeImage(0, drawImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        writer.updateSet(device, drawImageDescriptorSet);
    }

    // Add the descriptor allocator and layout destructors to the deletion queue
    mainDeletionQueue.pushFunction([=](){
        globalDescriptorAllocator.destroyPools(device);

        vkDestroyDescriptorSetLayout(device, drawImageDescriptorSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, displayTextureDescriptorSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, sceneDataDescriptorLayout, nullptr);
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

        frames[i].frameDescriptorAllocator = DescriptorAllocatorGrowable{};
        frames[i].frameDescriptorAllocator.init(device, 1000, framePoolSizes);

        // Pools in the frame descriptor allocators must be destroyed with the engine cleanup (not with frame cleanup)
        mainDeletionQueue.pushFunction([=]() {
            frames[i].frameDescriptorAllocator.destroyPools(device);
        });
    }
}

void VulkanEngine::m_initPasses()
{
    GLTFMetallicPass::Init(this);
}

void VulkanEngine::m_clearPassResources()
{
    GLTFMetallicPass::ClearResources(this);
}

void VulkanEngine::m_initMaterialLayouts()
{
    GLTFMetallicRoughnessMaterial::BuildMaterialLayout(this);
}

void VulkanEngine::m_clearMaterialLayouts()
{
    GLTFMetallicRoughnessMaterial::ClearMaterialLayout(device);
}

void VulkanEngine::m_initImgui()
{
    // 1: create descriptor pool for IMGUI
    // the size of the pool is very oversize, but it's copied from imgui demo  itself.
    VkDescriptorPoolSize poolSizes[] = { 
        { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 } 
    };

    VkDescriptorPoolCreateInfo poolInfo = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .pNext = nullptr};
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 1000;
    poolInfo.poolSizeCount = (uint32_t)std::size(poolSizes);
    poolInfo.pPoolSizes = poolSizes;

    VkDescriptorPool imguiPool; 
    VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &imguiPool));

    // 2. Initialize the ImGui Library
    // Initialize the core structures of ImGui
    ImGui::CreateContext();
    // Initialize ImGui for SDL
    ImGui_ImplSDL2_InitForVulkan(window);
    // Initialize ImGui for Vulkan
    ImGui_ImplVulkan_InitInfo initInfo = {};
    initInfo.Instance = instance;
    initInfo.PhysicalDevice = chosenGPU;
    initInfo.Device = device;
    initInfo.Queue = graphicsQueue;
    initInfo.DescriptorPool = imguiPool;
    initInfo.MinImageCount = 3;
    initInfo.ImageCount = 3;
    initInfo.UseDynamicRendering = true;

    // Dynamic rendering parameters for ImGui to use
    initInfo.PipelineRenderingCreateInfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .pNext = nullptr};
    initInfo.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    initInfo.PipelineRenderingCreateInfo.pColorAttachmentFormats = &swapchainImageFormat;

    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    ImGui_ImplVulkan_Init(&initInfo);

    ImGui_ImplVulkan_CreateFontsTexture();

    // Push the ImGui related destroy functions
    mainDeletionQueue.pushFunction([=](){
        ImGui_ImplVulkan_Shutdown();
        vkDestroyDescriptorPool(device, imguiPool, nullptr);
    });
}

void VulkanEngine::m_initDefaultData()
{
    // Default textures
    // 3 default textures 1 pixel each
    uint32_t white = glm::packUnorm4x8(glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
    whiteImage = createImage((void*)&white, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t grey = glm::packUnorm4x8(glm::vec4(0.66f, 0.66f, 0.66f, 1.0f));
    greyImage = createImage((void*)&grey, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t black = glm::packUnorm4x8(glm::vec4(0.0f, 0.0f, 0.0f, 0.0f));
    blackImage = createImage((void*)&black, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

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

    errorCheckerboardImage = createImage(pixels.data(), VkExtent3D{ 16, 16, 1 }, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    // Default samplers
    VkSamplerCreateInfo samplerInfo = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };

    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    vkCreateSampler(device, &samplerInfo, nullptr, &defaultSamplerNearest);

    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    vkCreateSampler(device, &samplerInfo, nullptr, &defaultSamplerLinear);

    mainDeletionQueue.pushFunction([=]() {
        destroyImage(whiteImage);
        destroyImage(greyImage);
        destroyImage(blackImage);
        destroyImage(errorCheckerboardImage);

        vkDestroySampler(device, defaultSamplerNearest, nullptr);
        vkDestroySampler(device, defaultSamplerLinear, nullptr);
    });

    // Default material data
    GLTFMetallicRoughnessMaterial::MaterialResources defaultMaterialResources;
    defaultMaterialResources.colorImage = whiteImage;
    defaultMaterialResources.colorSampler = defaultSamplerLinear;
    defaultMaterialResources.metalRoughnessImage = whiteImage;
    defaultMaterialResources.metalRoughnessSampler = defaultSamplerLinear;
    
    AllocatedBuffer materialConstantsBuffer = createBuffer(sizeof(GLTFMetallicRoughnessMaterial::MaterialConstants), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    // Write the buffer
    GLTFMetallicRoughnessMaterial::MaterialConstants* pMaterialConstantsBuffer = static_cast<GLTFMetallicRoughnessMaterial::MaterialConstants*>(materialConstantsBuffer.allocation->GetMappedData());
    pMaterialConstantsBuffer->colorFactors = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    pMaterialConstantsBuffer->metalRoughnessFactors = glm::vec4(1.0f, 0.5f, 0.0f, 0.0f);

    mainDeletionQueue.pushFunction([=]() {
        destroyBuffer(materialConstantsBuffer);
    });

    defaultMaterialResources.dataBuffer = materialConstantsBuffer.buffer;
    defaultMaterialResources.dataBufferOffset = 0;

    defaultMaterialInstance = GLTFMetallicRoughnessMaterial::CreateInstance(device, MaterialPass::Opaque, defaultMaterialResources, globalDescriptorAllocator);
}

void VulkanEngine::m_initGlobalSceneBuffer()
{
    for(int i = 0; i < FRAME_OVERLAP; ++i)
    {
        // Allocate a new uniform buffer for scene data (allocating on VRAM that CPU can write to directly. It is limited but it is perfect for allocating reasonable amounts that are dynamic)
        gpuSceneDataBuffer[i] = createBuffer(sizeof(GPUSceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        mainDeletionQueue.pushFunction([=]() {
            destroyBuffer(gpuSceneDataBuffer[i]);
        });

        // Create a descriptor set for the uniform data
        sceneDescriptorSet[i] = globalDescriptorAllocator.allocate(device, sceneDataDescriptorLayout);
        DescriptorWriter writer;
        writer.writeBuffer(0, gpuSceneDataBuffer[i].buffer, sizeof(GPUSceneData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        writer.updateSet(device, sceneDescriptorSet[i]);
    }
}

void VulkanEngine::m_initCamera(glm::vec3 position, float pitch, float yaw)
{
    mainCamera.velocity = glm::vec3(0.0f);
    mainCamera.position = position;
    mainCamera.pitch = pitch;
    mainCamera.yaw = yaw;
}

void VulkanEngine::m_loadSceneData()
{
    std::string structurePath = "../../assets/structure.glb";
    auto loadedStructureScene = loadGltf(this, structurePath);
    assert(loadedStructureScene.has_value());
    loadedScenes["structure"] = loadedStructureScene.value();
}
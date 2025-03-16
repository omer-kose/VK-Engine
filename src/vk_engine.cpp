#include "vk_engine.h"

#include <chrono>
#include <thread>

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_initializers.h>
#include <vk_types.h>
#include <vk_images.h>
#include <vk_pipelines.h>
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
    m_initPipelines();

    m_initImgui();

    m_initDefaultData();

    // everything went fine
    isInitialized = true;
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

        // Clear the loaded meshes
        for(auto& mesh : testMeshes)
        {
            destroyBuffer(mesh->meshBuffers.vertexBuffer);
            destroyBuffer(mesh->meshBuffers.indexBuffer);
        }

        metallicRoughnessMaterial.clearResources(device);
        
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

    // Transition the draw image into writeable mode before rendering
    vkutil::transitionImage(cmd, drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    drawBackground(cmd);

    // When rendering geometry we need to use COLOR_ATTACHMENT_OPTIMAL as it is the most optimal layout for rendering with graphics pipeline
    vkutil::transitionImage(cmd, drawImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkutil::transitionImage(cmd, depthImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    
    drawGeometry(cmd);

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

void VulkanEngine::drawBackground(VkCommandBuffer cmd)
{
    ComputeEffect& effect = backgroundEffects[currentBackgroundEffect];
    // bind the gradient drawing compute pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, effect.pipeline);

    // Bind the descriptor sets
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gradientPipelineLayout, 0, 1, &drawImageDescriptorSet, 0, nullptr);

    // Push Constants
    vkCmdPushConstants(cmd, gradientPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePushConstants), &effect.pushConstants);

    // dispatch the compute shader. We are using 16x16 workgroup size
    vkCmdDispatch(cmd, std::ceil(drawExtent.width / 16), std::ceil(drawExtent.height / 16), 1);
}

void VulkanEngine::drawGeometry(VkCommandBuffer cmd)
{
    // Begin a renderpass connected to the draw image
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(drawImage.imageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo depthAttachment = vkinit::depth_attachment_info(depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    VkRenderingInfo renderInfo = vkinit::rendering_info(drawExtent, &colorAttachment, &depthAttachment);
    vkCmdBeginRendering(cmd, &renderInfo);
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeline);

    // Set dynamic viewport and scissor (dynamic states must be set after a pipeline with dynamic states bound. After setting once, no need for setting for subsequent pipeline bindings (with dynamic states))
    VkViewport viewport = {};
    viewport.x = 0;
    viewport.y = 0;
    viewport.width = drawExtent.width;
    viewport.height = drawExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor = {};
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent.width = drawExtent.width;
    scissor.extent.height = drawExtent.height;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Bind a texture
    VkDescriptorSet textureSet = getCurrentFrame().frameDescriptorAllocator.allocate(device, displayTextureDescriptorSetLayout);
    {
        DescriptorWriter writer;
        writer.writeImage(0, errorCheckerboardImage.imageView, defaultSamplerNearest, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        writer.updateSet(device, textureSet);
    }

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipelineLayout, 0, 1, &textureSet, 0, nullptr);

    // Draw the test mesh (basic.glb has 3 meshes inside, drawing the 3rd mesh for the test)
    GPUDrawPushConstants pushConstants;
    glm::mat4 view = glm::translate(glm::vec3(0.0f, 0.0f, -5.0f));
    glm::mat4 projection = glm::perspective(glm::radians(70.0f), (float)drawExtent.width / (float)drawExtent.height, 0.1f, 10000.0f);
    projection[1][1] *= -1.0f;
    pushConstants.worldMatrix = projection * view;
    pushConstants.vertexBuffer = testMeshes[2]->meshBuffers.vertexBufferAddress;
    vkCmdPushConstants(cmd, meshPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GPUDrawPushConstants), &pushConstants);
    vkCmdBindIndexBuffer(cmd, testMeshes[2]->meshBuffers.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);;
    vkCmdDrawIndexed(cmd, testMeshes[2]->surfaces[0].count, 1, testMeshes[2]->surfaces[0].startIndex, 0, 0);

    vkCmdEndRendering(cmd);
}

void VulkanEngine::drawImgui(VkCommandBuffer cmd, VkImageView targetImageView)
{
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(targetImageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingInfo renderInfo = vkinit::rendering_info(swapchainExtent, &colorAttachment, nullptr);

    vkCmdBeginRendering(cmd, &renderInfo);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRendering(cmd);
}

void VulkanEngine::run()
{
    SDL_Event e;
    bool bQuit = false;

    // main loop
    while(!bQuit) 
    {
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

        if(ImGui::Begin("background"))
        {
            ImGui::SliderFloat("Render Scale", &renderScale, 0.3f, 1.0f);

            ComputeEffect& selected = backgroundEffects[currentBackgroundEffect];
            
            ImGui::Text("Selected effect: ", selected.name);

            ImGui::SliderInt("Effect index", &currentBackgroundEffect, 0, backgroundEffects.size() - 1);

            ImGui::InputFloat4("data1", (float*)&selected.pushConstants.data1);
            ImGui::InputFloat4("data2", (float*)&selected.pushConstants.data2);
            ImGui::InputFloat4("data3", (float*)&selected.pushConstants.data3);
            ImGui::InputFloat4("data4", (float*)&selected.pushConstants.data4);
        }
        ImGui::End();

        // Make ImGui calculate internal draw structures
        ImGui::Render();

        draw();
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

        vkutil::transitionImage(cmd, newImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
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

void VulkanEngine::m_initPipelines()
{
    // Compute Pipelines
    m_initBackgroundPipelines();
    // Graphics Pipelines
    m_initMeshPipeline();
    metallicRoughnessMaterial.buildPipelines(this);
}

void VulkanEngine::m_initBackgroundPipelines()
{
    VkPipelineLayoutCreateInfo computeLayout{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pNext = nullptr};
    computeLayout.pSetLayouts = &drawImageDescriptorSetLayout;
    computeLayout.setLayoutCount = 1;

    // Push Constants
    VkPushConstantRange pushConstants{};
    pushConstants.offset = 0;
    pushConstants.size = sizeof(ComputePushConstants);
    pushConstants.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    computeLayout.pushConstantRangeCount = 1;
    computeLayout.pPushConstantRanges = &pushConstants;

    VK_CHECK(vkCreatePipelineLayout(device, &computeLayout, nullptr, &gradientPipelineLayout));

    VkShaderModule gradientShader;
    if(!vkutil::loadShaderModule(device, "../../shaders/gradient_color.comp.spv", &gradientShader))
    {
        fmt::print("Error when building the compute shader \n");
    }

    VkShaderModule skyShader;
    if(!vkutil::loadShaderModule(device, "../../shaders/sky.comp.spv", &skyShader))
    {
        fmt::print("Error when building the compute shader \n");
    }

    VkPipelineShaderStageCreateInfo stageInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = nullptr};
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = gradientShader;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo computePipelineCreateInfo{.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .pNext = nullptr};
    computePipelineCreateInfo.layout = gradientPipelineLayout;
    computePipelineCreateInfo.stage = stageInfo;
    
    // Create gradient background pipeline
    ComputeEffect gradient;
    gradient.name = "gradient";
    gradient.pipelineLayout = gradientPipelineLayout;
    gradient.pushConstants = {};

    // default colors
    gradient.pushConstants.data1 = glm::vec4(1.0, 0.0, 0.0, 1.0);
    gradient.pushConstants.data2 = glm::vec4(0.0, 0.0, 1.0, 1.0);
    
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &gradient.pipeline));

    // Create sky background pipeline
    // The only thing differs between two pipelines is the shader module
    computePipelineCreateInfo.stage.module = skyShader;

    ComputeEffect sky;
    sky.name = "sky";
    sky.pipelineLayout = gradientPipelineLayout;
    sky.pushConstants = {};
    // default sky parameters
    sky.pushConstants.data1 = glm::vec4(0.1, 0.2, 0.4, 0.97);
    
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &sky.pipeline));

    // Store the created effects
    backgroundEffects.push_back(gradient);
    backgroundEffects.push_back(sky);

    // We don't need the shader module after binding it into the pipeline
    vkDestroyShaderModule(device, gradientShader, nullptr);
    vkDestroyShaderModule(device, skyShader, nullptr);

    mainDeletionQueue.pushFunction([=]() {
        vkDestroyPipelineLayout(device, gradientPipelineLayout, nullptr);
        vkDestroyPipeline(device, backgroundEffects[0].pipeline, nullptr);
        vkDestroyPipeline(device, backgroundEffects[1].pipeline, nullptr);
    });
}

void VulkanEngine::m_initMeshPipeline()
{
    // Load the shaders
    VkShaderModule meshVertexShader;
    if(!vkutil::loadShaderModule(device, "../../shaders/colored_mesh.vert.spv", &meshVertexShader))
    {
        fmt::println("Error while building the mesh vertex shader module");
    }
    else
    {
        fmt::println("Mesh vertex shader successfully loaded");
    }

    VkShaderModule meshFragmentShader;
    if(!vkutil::loadShaderModule(device, "../../shaders/display_texture.frag.spv", &meshFragmentShader))
    {
        fmt::println("Error while building the mesh fragment shader module");
    }
    else
    {
        fmt::println("Mesh fragment shader successfully loaded");
    }

    // Pipeline layout
    VkPushConstantRange bufferRange{};
    bufferRange.offset = 0;
    bufferRange.size = sizeof(GPUDrawPushConstants);
    bufferRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = vkinit::pipeline_layout_create_info();
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &bufferRange;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &displayTextureDescriptorSetLayout;
    VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &meshPipelineLayout));

    // Build the pipeline
    PipelineBuilder pipelineBuilder;
    // Use the triangle pipeline layout we created
    pipelineBuilder.pipelineLayout = meshPipelineLayout;
    // Connect the vertex and fragment shaders to the pipeline
    pipelineBuilder.setShaders(meshVertexShader, meshFragmentShader);
    // It will draw triangles
    pipelineBuilder.setInputTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    // Filled triangles
    pipelineBuilder.setPolygonMode(VK_POLYGON_MODE_FILL);
    // No backface culling
    pipelineBuilder.setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    // No multisampling
    pipelineBuilder.setMultiSamplingNone();
    // No blending
    pipelineBuilder.disableBlending();
    // Enable depth test
    pipelineBuilder.enableDepthTest(true, VK_COMPARE_OP_LESS_OR_EQUAL);

    // Connect the image formats
    pipelineBuilder.setColorAttachmentFormat(drawImage.imageFormat);
    pipelineBuilder.setDepthFormat(depthImage.imageFormat);

    // Finally build the pipeline
    meshPipeline = pipelineBuilder.buildPipeline(device);

    // No need for shader modules ones they are bound to the pipeline
    vkDestroyShaderModule(device, meshVertexShader, nullptr);
    vkDestroyShaderModule(device, meshFragmentShader, nullptr);

    mainDeletionQueue.pushFunction([=]() {
        vkDestroyPipelineLayout(device, meshPipelineLayout, nullptr);
        vkDestroyPipeline(device, meshPipeline, nullptr);
    });
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
    // Default meshes
    testMeshes = loadGltfMeshes(this, "../../assets/basicmesh.glb").value();

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

    defaultMaterial = metallicRoughnessMaterial.createInstance(device, MaterialPass::Opaque, defaultMaterialResources, globalDescriptorAllocator);
}

void GLTFMetallicRoughnessMaterial::buildPipelines(VulkanEngine* engine)
{
    // Load the shaders
    VkShaderModule meshVertexShader;
    if(!vkutil::loadShaderModule(engine->device, "../../shaders/mesh.vert.spv", &meshVertexShader))
    {
        fmt::println("Error when building the mesh vertex shader");
    }

    VkShaderModule meshFragmentShader;
    if(!vkutil::loadShaderModule(engine->device, "../../shaders/mesh.frag.spv", &meshFragmentShader))
    {
        fmt::println("Error when building the mesh fragment shader");
    }

    // Set push constant range
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(GPUDrawPushConstants);
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    // Set descriptor sets
    // Material set (set 1)
    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    layoutBuilder.addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    layoutBuilder.addBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    materialLayout = layoutBuilder.build(engine->device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

    // 2 sets: 0 -> Scene Descriptor Set, 1 -> Material Descriptor Set
    VkDescriptorSetLayout layouts [] = { engine->sceneDataDescriptorLayout, materialLayout };

    // Mesh pipeline layout
    VkPipelineLayoutCreateInfo meshLayoutInfo = vkinit::pipeline_layout_create_info();
    meshLayoutInfo.pushConstantRangeCount = 1;
    meshLayoutInfo.pPushConstantRanges = &pushConstantRange;
    meshLayoutInfo.setLayoutCount = 2;
    meshLayoutInfo.pSetLayouts = layouts;
    
    VkPipelineLayout meshPipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(engine->device, &meshLayoutInfo, nullptr, &meshPipelineLayout));
    
    // Both pipelines have the same layout
    opaquePipeline.layout = meshPipelineLayout;
    transparentPipeline.layout = meshPipelineLayout;

    // Build the pipelines
    PipelineBuilder pipelineBuilder;
    pipelineBuilder.setShaders(meshVertexShader, meshFragmentShader);
    pipelineBuilder.setInputTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.setPolygonMode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE); // TODO: Try VK_FRONT_FACE_COUNTER_CLOCKWISE and make it compatible to that.
    pipelineBuilder.setMultiSamplingNone();
    pipelineBuilder.disableBlending();
    pipelineBuilder.enableDepthTest(true, VK_COMPARE_OP_LESS_OR_EQUAL);

    // Render format
    pipelineBuilder.setColorAttachmentFormat(engine->drawImage.imageFormat);
    pipelineBuilder.setDepthFormat(engine->depthImage.imageFormat);
    
    pipelineBuilder.pipelineLayout = meshPipelineLayout;
    // Opaque Pipeline
    opaquePipeline.pipeline = pipelineBuilder.buildPipeline(engine->device);
    
    // Transparent variant
    pipelineBuilder.enableBlendingAdditive();
    pipelineBuilder.enableDepthTest(false, VK_COMPARE_OP_LESS_OR_EQUAL);
    transparentPipeline.pipeline = pipelineBuilder.buildPipeline(engine->device);

    // ShaderModules are not needed anymore
    vkDestroyShaderModule(engine->device, meshVertexShader, nullptr);
    vkDestroyShaderModule(engine->device, meshFragmentShader, nullptr);
}

void GLTFMetallicRoughnessMaterial::clearResources(VkDevice device)
{
    vkDestroyDescriptorSetLayout(device, materialLayout, nullptr);
    // both opaque and transparent pipelines has the same layout vulkan handle so destroying one is enough
    vkDestroyPipelineLayout(device, opaquePipeline.layout, nullptr);

    vkDestroyPipeline(device, transparentPipeline.pipeline, nullptr);
    vkDestroyPipeline(device, opaquePipeline.pipeline, nullptr);
}

MaterialInstance GLTFMetallicRoughnessMaterial::createInstance(VkDevice device, MaterialPass pass, const MaterialResources& resources, DescriptorAllocatorGrowable& descriptorAllocator)
{
    MaterialInstance matData;
    matData.passType = pass;
    if(pass == MaterialPass::Opaque)
    {
        matData.pipeline = &opaquePipeline;
    }
    else
    {
        matData.pipeline = &transparentPipeline;
    }

    matData.materialSet = descriptorAllocator.allocate(device, materialLayout);

    writer.clear();
    writer.writeBuffer(0, resources.dataBuffer, sizeof(MaterialConstants), resources.dataBufferOffset, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.writeImage(1, resources.colorImage.imageView, resources.colorSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.writeImage(2, resources.metalRoughnessImage.imageView, resources.metalRoughnessSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.updateSet(device, matData.materialSet);

    return matData;
}

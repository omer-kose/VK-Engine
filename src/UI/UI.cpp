#include "UI.h"

#include <Core/vk_renderer.h>
#include <Core/vk_initializers.h>

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

#include "SDL_events.h"

void SK::UI::init(UI* ui, SK::VkRenderer::Renderer* renderer)
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

    VkDescriptorPoolCreateInfo poolInfo = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .pNext = nullptr };
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 1000;
    poolInfo.poolSizeCount = (uint32_t)std::size(poolSizes);
    poolInfo.pPoolSizes = poolSizes;

    VkDescriptorPool imguiPool;
    VK_CHECK(vkCreateDescriptorPool(renderer->device, &poolInfo, nullptr, &imguiPool));

    // 2. Initialize the ImGui Library
    // Initialize the core structures of ImGui
    ImGui::CreateContext();
    // Initialize ImGui for SDL
    ImGui_ImplSDL2_InitForVulkan(renderer->window);
    // Initialize ImGui for Vulkan
    ImGui_ImplVulkan_InitInfo initInfo = {};
    initInfo.Instance = renderer->instance;
    initInfo.PhysicalDevice = renderer->chosenGPU;
    initInfo.Device = renderer->device;
    initInfo.Queue = renderer->graphicsQueue;
    initInfo.DescriptorPool = imguiPool;
    initInfo.MinImageCount = 3;
    initInfo.ImageCount = 3;
    initInfo.UseDynamicRendering = true;

    // Dynamic rendering parameters for ImGui to use
    initInfo.PipelineRenderingCreateInfo = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .pNext = nullptr };
    initInfo.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    initInfo.PipelineRenderingCreateInfo.pColorAttachmentFormats = &renderer->swapchainImageFormat;

    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    ImGui_ImplVulkan_Init(&initInfo);

    ImGui_ImplVulkan_CreateFontsTexture();

    // UI will destroy its own resources
    ui->deletionQueue.pushFunction([=]() {
        ImGui_ImplVulkan_Shutdown();
        vkDestroyDescriptorPool(renderer->device, imguiPool, nullptr);
    });

    // Register UI draw to renderer's overlay passes
    SK::VkRenderer::OverlayPass pass;
    pass.draw = &draw;
    SK::VkRenderer::registerOverlayPass(renderer, pass);

    ui->isInitialized = true;
}

void SK::UI::processSDLEvents(const SDL_Event& e)
{
    ImGui_ImplSDL2_ProcessEvent(&e);
}

void SK::UI::draw(SK::VkRenderer::PassContext* ctx)
{
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(ctx->targetImageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingInfo renderInfo = vkinit::rendering_info(ctx->imageExtent, &colorAttachment, nullptr);

    vkCmdBeginRendering(ctx->cmd, &renderInfo);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), ctx->cmd);
    vkCmdEndRendering(ctx->cmd);
}

void SK::UI::shutdown(UI* ui)
{
    if(ui->isInitialized)
    {
        ui->deletionQueue.flush();
        ui->isInitialized = true;
    }
}

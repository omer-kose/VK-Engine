#pragma once 
#include <vk_types.h>

class PipelineBuilder
{
public:
    PipelineBuilder();
    void clear();
    VkPipeline buildPipeline(VkDevice device);

    void setShaders(VkShaderModule vertexShader, VkShaderModule fragmentShader);
    void setInputTopology(VkPrimitiveTopology topology);
    void setPolygonMode(VkPolygonMode polygonMode);
    void setCullMode(VkCullModeFlags cullMode, VkFrontFace frontFace);
    void setMultiSamplingNone();
    void disableBlending();
    void enableBlendingAdditive();
    void enableBlendingAlphaBlend();
    void setColorAttachmentFormat(VkFormat format);
    void setDepthFormat(VkFormat format);
    void disableDepthTest();
    void enableDepthTest(bool depthWriteEnable, VkCompareOp compareOp);
public:
	std::vector<VkPipelineShaderStageCreateInfo> shaderStages;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly;
    VkPipelineRasterizationStateCreateInfo rasterizer;
    VkPipelineColorBlendAttachmentState colorBlendAttachment;
    VkPipelineMultisampleStateCreateInfo multisampling;
    VkPipelineLayout pipelineLayout;
    VkPipelineDepthStencilStateCreateInfo depthStencil;
    VkPipelineRenderingCreateInfo renderInfo;
    VkFormat colorAttachmentformat;
};

namespace vkutil 
{
	bool loadShaderModule(VkDevice device, const char* filePath, VkShaderModule* outShaderModule);
};
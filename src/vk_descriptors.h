#pragma once

#include <vk_types.h>

struct DescriptorLayoutBuilder
{
	void addBinding(uint32_t bindingSlot, VkDescriptorType type);
	void clear();
	VkDescriptorSetLayout build(VkDevice device, VkShaderStageFlags shaderStages, void* pNext = nullptr, VkDescriptorSetLayoutCreateFlags flags = 0);

	std::vector<VkDescriptorSetLayoutBinding> bindings;
};

struct DescriptorWriter
{
	// Storing infos for keeping the pointers to them valid in writes. VkWriteDescriptorSet holds a pointer to the descriptor information to be updated. So, those pointers must be valid.
	std::deque<VkDescriptorImageInfo> imageInfos;
	std::deque<VkDescriptorBufferInfo> bufferInfos;

	std::vector<VkWriteDescriptorSet> writes;

	void writeImage(int binding, VkImageView imageView, VkSampler sampler, VkImageLayout layout, VkDescriptorType type);
	void writeBuffer(int binding, VkBuffer buffer, size_t size, size_t offset, VkDescriptorType type);

	void clear();
	void updateSet(VkDevice device, VkDescriptorSet set);
};

struct DescriptorAllocator
{
	struct PoolSize
	{
		VkDescriptorType type;
		uint32_t count;
	};

	void initPool(VkDevice device, uint32_t maxSets, std::span<PoolSize> poolSizes);
	void clearDescriptors(VkDevice device);
	void destroyPool(VkDevice device);

	VkDescriptorSet allocate(VkDevice device, VkDescriptorSetLayout layout);

	VkDescriptorPool pool;
};

struct DescriptorAllocatorGrowable
{
public:
	struct PoolSize
	{
		VkDescriptorType type;
		uint32_t count;
	};

	void init(VkDevice device, uint32_t maxSets, std::span<PoolSize> poolSizes);
	void clearPools(VkDevice device);
	void destroyPools(VkDevice device);

	VkDescriptorSet allocate(VkDevice device, VkDescriptorSetLayout layout);

private:
	VkDescriptorPool getPool(VkDevice device);
	VkDescriptorPool createPool(VkDevice device, uint32_t maxSets, std::span<PoolSize> poolSizes);

	std::vector<PoolSize> m_poolSizes;
	std::vector<VkDescriptorPool> m_fullPools;
	std::vector<VkDescriptorPool> m_readyPools;
	uint32_t m_maxSetsPerPool;
};
#include <Core/vk_descriptors.h>

void DescriptorLayoutBuilder::addBinding(uint32_t bindingSlot, VkDescriptorType type)
{
	VkDescriptorSetLayoutBinding newBinding{};
	newBinding.binding = bindingSlot;
	newBinding.descriptorCount = 1;
	newBinding.descriptorType = type;
	
	bindings.push_back(newBinding);
}

void DescriptorLayoutBuilder::clear()
{
	bindings.clear();
}

VkDescriptorSetLayout DescriptorLayoutBuilder::build(VkDevice device, VkShaderStageFlags shaderStages, VkDescriptorSetLayoutCreateFlags flags, void* pNext)
{
	for(auto& b : bindings)
	{
		b.stageFlags |= shaderStages;
	}

	VkDescriptorSetLayoutCreateInfo info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
	info.pNext = pNext;

	info.bindingCount = (uint32_t)bindings.size();
	info.pBindings = bindings.data();
	info.flags = flags;

	VkDescriptorSetLayout setLayout;
	VK_CHECK(vkCreateDescriptorSetLayout(device, &info, nullptr, &setLayout));

	return setLayout;
}

void DescriptorWriter::writeImage(int binding, VkImageView imageView, VkSampler sampler, VkImageLayout layout, VkDescriptorType type)
{
	VkDescriptorImageInfo& info = imageInfos.emplace_back(VkDescriptorImageInfo{
		.sampler = sampler,
		.imageView = imageView,
		.imageLayout = layout
	});

	VkWriteDescriptorSet write = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = nullptr };
	write.dstBinding = binding;
	write.dstSet = VK_NULL_HANDLE; // left empty for now until the actual update time comes
	write.descriptorCount = 1;
	write.descriptorType = type;
	write.pImageInfo = &info;

	writes.push_back(write);
}

void DescriptorWriter::writeBuffer(int binding, VkBuffer buffer, size_t size, size_t offset, VkDescriptorType type)
{
	VkDescriptorBufferInfo& info = bufferInfos.emplace_back(VkDescriptorBufferInfo{
		.buffer = buffer,
		.offset = offset,
		.range = size
	});

	VkWriteDescriptorSet write = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = nullptr };
	write.dstBinding = binding;
	write.dstSet = VK_NULL_HANDLE; // left empty for now until the actual update time comes
	write.descriptorCount = 1;
	write.descriptorType = type;
	write.pBufferInfo = &info;

	writes.push_back(write);
}

void DescriptorWriter::clear()
{
	imageInfos.clear();
	bufferInfos.clear();
	writes.clear();
}

void DescriptorWriter::updateSet(VkDevice device, VkDescriptorSet set)
{
	for(VkWriteDescriptorSet& write : writes)
	{
		write.dstSet = set;
	}

	vkUpdateDescriptorSets(device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
}


void DescriptorAllocator::initPool(VkDevice device, uint32_t maxSets, std::span<PoolSize> poolSizes)
{
	/*
		VkDescriptorPoolSize describes how many individual descriptors of a given type can be allocated from the pool accross all the sets. 
		Note that this is not the total number of descriptors per set. This is the total number of allocations can be done per descriptor type from this pool. 
		So, the maxSets field in the VkDescriptorPoolCreateInfo does not multiply those sizes with the maxSets, it only specifies how many sets in total can be allocated from this pool.
		For example, if we are going to have 2 sets in total with a uniform buffer each, the descriptorCount for the uniform buffer type in VkDescriptorPoolSize must be 2. 
	*/
	std::vector<VkDescriptorPoolSize> descriptorPoolSizes;
	for(const PoolSize& poolSize : poolSizes)
	{
		descriptorPoolSizes.push_back(VkDescriptorPoolSize{
			.type = poolSize.type,
			.descriptorCount = poolSize.count * maxSets
		});
	}

	VkDescriptorPoolCreateInfo poolInfo = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .pNext = nullptr};
	poolInfo.flags = 0;
	poolInfo.maxSets = maxSets;
	poolInfo.poolSizeCount = (uint32_t)descriptorPoolSizes.size();
	poolInfo.pPoolSizes = descriptorPoolSizes.data();

	VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool));
}

void DescriptorAllocator::clearDescriptors(VkDevice device)
{
	// Resettign descriptor pool destroys all the descriptors allocated from this pool and reset backs to pool into its initial state
	vkResetDescriptorPool(device, pool, 0);
}

void DescriptorAllocator::destroyPool(VkDevice device)
{
	vkDestroyDescriptorPool(device, pool, nullptr);
}

VkDescriptorSet DescriptorAllocator::allocate(VkDevice device, VkDescriptorSetLayout layout)
{
	VkDescriptorSetAllocateInfo allocInfo = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .pNext = nullptr};
	allocInfo.descriptorPool = pool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &layout;

	VkDescriptorSet set;
	VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &set));
	
	return set;
}

void DescriptorAllocatorGrowable::init(VkDevice device, uint32_t maxSets, std::span<PoolSize> poolSizes)
{
	m_poolSizes.clear();

	for(PoolSize poolSize : poolSizes)
	{
		m_poolSizes.push_back(poolSize);
	}

	VkDescriptorPool newPool = createPool(device, maxSets, poolSizes);
	m_maxSetsPerPool = maxSets * 1.5f; // increase the capability of the next pool to be allocated
	m_readyPools.push_back(newPool);
}

void DescriptorAllocatorGrowable::clearPools(VkDevice device)
{
	for(auto  p : m_readyPools)
	{
		vkResetDescriptorPool(device, p, 0);
	}

	for(auto p : m_fullPools)
	{
		vkResetDescriptorPool(device, p, 0);
		m_readyPools.push_back(p);
	}

	m_fullPools.clear();
}

void DescriptorAllocatorGrowable::destroyPools(VkDevice device)
{
	for(auto p : m_readyPools)
	{
		vkDestroyDescriptorPool(device, p, nullptr);
	}

	m_readyPools.clear();

	for(auto p : m_fullPools)
	{
		vkDestroyDescriptorPool(device, p, nullptr);
	}

	m_fullPools.clear();
}

VkDescriptorSet DescriptorAllocatorGrowable::allocate(VkDevice device, VkDescriptorSetLayout layout)
{
	VkDescriptorPool poolToUse = getPool(device);

	VkDescriptorSetAllocateInfo allocInfo = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .pNext = nullptr };
	allocInfo.descriptorPool = poolToUse;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &layout;

	VkDescriptorSet set;
	VkResult result = vkAllocateDescriptorSets(device, &allocInfo, &set);

	// If allocation is failed try again
	if(result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL)
	{
		m_fullPools.push_back(poolToUse);
		// Get a new pool
		poolToUse = getPool(device);
		allocInfo.descriptorPool = poolToUse;

		// if allocation is failed for the second time just crash
		VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &set));
	}

	m_readyPools.push_back(poolToUse);
	return set;
}	

VkDescriptorPool DescriptorAllocatorGrowable::getPool(VkDevice device)
{
	VkDescriptorPool newPool;
	if(m_readyPools.size() != 0)
	{
		newPool = m_readyPools.back();
		m_readyPools.pop_back();
	}
	else
	{
		// need to create a new pool
		newPool = createPool(device, m_maxSetsPerPool, m_poolSizes);

		// in each creation increase the capability of the following pools by increasing the maxSetsPerPool (which in turn increases the number of descriptors that can be allocated from the newer pools to be created later)
		m_maxSetsPerPool = m_maxSetsPerPool * 1.5f;
		if(m_maxSetsPerPool > 4092)
		{
			m_maxSetsPerPool = 4092; // hardcoded limit to avoid pools growing too much
		}
	}

	return newPool;
}

VkDescriptorPool DescriptorAllocatorGrowable::createPool(VkDevice device, uint32_t maxSets, std::span<PoolSize> poolSizes)
{
	std::vector<VkDescriptorPoolSize> descriptorPoolSizes;
	for(const PoolSize& poolSize : poolSizes)
	{
		descriptorPoolSizes.push_back(VkDescriptorPoolSize{
			.type = poolSize.type,
			.descriptorCount = poolSize.count * maxSets
		});
	}
	
	VkDescriptorPoolCreateInfo poolInfo = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .pNext = nullptr };
	poolInfo.flags = 0;
	poolInfo.maxSets = maxSets;
	poolInfo.poolSizeCount = (uint32_t)poolSizes.size();
	poolInfo.pPoolSizes = descriptorPoolSizes.data();

	VkDescriptorPool newPool;
	VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &newPool));
	return newPool;
}

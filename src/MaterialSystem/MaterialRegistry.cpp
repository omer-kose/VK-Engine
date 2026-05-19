#include "MaterialRegistry.h"

uint32_t SK::Material::registerInstance(MaterialRegistry* materialRegistry, Instance&& instance)
{
    uint32_t idx = static_cast<uint32_t>(materialRegistry->instances.size());
    materialRegistry->instances.push_back(std::move(instance));
    return idx;
}

void SK::Material::clearMaterialRegistry(MaterialRegistry* materialRegistry)
{
    materialRegistry->instances.clear();
}

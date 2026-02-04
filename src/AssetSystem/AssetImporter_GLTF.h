#pragma once

#include <string_view>
#include "ImportedAsset.h"

namespace SK::Asset
{
    bool importGLTF(std::string_view filePath, ImportedAsset* outAsset);
}
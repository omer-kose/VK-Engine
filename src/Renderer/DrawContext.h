#pragma once

#include <vector>
#include "DrawPacket.h"

namespace SK::Renderer
{
    struct DrawContext
    {
        std::vector<DrawPacket> opaque;
        std::vector<DrawPacket> transparent;

        void clear()
        {
            opaque.clear();
            transparent.clear();
        }
    };
}
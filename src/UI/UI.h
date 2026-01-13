/*
	UI Layer
*/
#pragma once

#include <Util/DeletionQueue.h>

// Forward declarations
union SDL_Event;

namespace SK::VkRenderer
{
	struct Renderer;
	struct PassContext; // for ImGui overlay draw pass
};

namespace SK::UI
{
	struct UI
	{
		SK::Util::DeletionQueue deletionQueue;
		bool isInitialized = false;
	};

	void init(UI* ui, SK::VkRenderer::Renderer* renderer);
	void processSDLEvents(const SDL_Event& e);
	void shutdown(UI* ui);

	// Registered to the Renderer's Overlay passes.
	void draw(SK::VkRenderer::PassContext* ctx);
};
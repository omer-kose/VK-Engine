/*
	UI Layer
*/
#pragma once

#include <Util/DeletionQueue.h>

// Forward declarations
union SDL_Event;

namespace SK::VkRendererBackend
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

	void init(UI* ui, SK::VkRendererBackend::Renderer* renderer);
	void processSDLEvents(const SDL_Event& e);
	void beginFrame();
	void endFrame();
	void shutdown(UI* ui);

	// Registered to the Renderer's Overlay passes.
	void draw(SK::VkRendererBackend::PassContext* ctx);
};
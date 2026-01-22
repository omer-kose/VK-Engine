/*
	UI Layer
*/
#pragma once

#include <Util/DeletionQueue.h>

// Forward declarations
union SDL_Event;

namespace SK::VkRendererBackend
{
	struct State;
	struct PassContext; // for ImGui overlay draw pass
};

namespace SK::UI
{
	struct State
	{
		SK::Util::DeletionQueue deletionQueue;
		bool isInitialized = false;
	};

	void init(State* ui, SK::VkRendererBackend::State* vkRendererBackend);
	void processSDLEvents(const SDL_Event& e);
	void beginFrame();
	void endFrame();
	void shutdown(State* ui);

	// Registered to the RendererBackend's Overlay passes.
	void draw(SK::VkRendererBackend::PassContext* ctx);
};
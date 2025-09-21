#ifndef WINDOWS_T
#define WINDOWS_T


#define WM_CUDA_SAMPLE_MSG_DESTROY          (WM_APP + 1)

#include "framework.h"

typedef struct {
    HWND windowsHandle;
    bool isRunning;
	wchar_t className[256];
} Windows_t;

long windows_create(Windows_t* parent, HINSTANCE hInstance, WNDPROC windowsProc);
long windowsGetEvents(Windows_t* parent);
void windowsDestroy(Windows_t* parent);


#endif// WINDOWS_T





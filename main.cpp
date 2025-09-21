#include "framework.h"
#include "windows_handler.h"

#define MAX_LOADSTRING 100

const wchar_t windowClassName[] = L"CUDA_SAMPLE";

#define WM_CUDA_SAMPLE_MSG_LOAD_IMAGE   (WM_USER + 1)


LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam){

    Windows_t* pWin = reinterpret_cast<Windows_t*>(
        GetWindowLongPtr(hWnd, GWLP_USERDATA)
        );

    switch (message) {
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(pWin->windowsHandle, &ps);
            // TODO: Add any drawing code that uses hdc here...
            EndPaint(pWin->windowsHandle, &ps);
            break;
        }

        case WM_DESTROY: {
            PostQuitMessage(0);
            break;
        }

        case WM_CUDA_SAMPLE_MSG_DESTROY: {
            Windows_t* pWin = reinterpret_cast<Windows_t*>(lParam);
            if (pWin) {
                pWin->isRunning = false;
                windowsDestroy(pWin);
            }
        }

        default:
            return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}


int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
    _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPWSTR    lpCmdLine,
    _In_ int       nCmdShow) {

    Windows_t win = {0};
	wcscpy_s(win.className, L"CUDA_SAMPLE");
	windows_create(&win, hInstance, WndProc);

    while (win.isRunning) {
		windowsGetEvents(&win);
    }

    return 1;
}
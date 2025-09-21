#ifndef BITMAP_T_H
#define BITMAP_T_H

#include <windows.h>
#include <cstdint>

typedef struct {
	HBITMAP hBitmap;
    uint32_t width;
    uint32_t height;
    uint8_t bitDepth;
    uint8_t* data;
    size_t lineStride;
} Bitmap_t;

long windowsBitmapCreate(HWND windowsHandle, Bitmap_t* self, uint32_t width, uint32_t height, uint8_t bitDepth);
size_t calcLineStride(Bitmap_t* btmp);
void drawBitmap(HWND handle, Bitmap_t* btmp, int32_t x, int32_t y);
void destroyBitmap(Bitmap_t* btmp);
void sexPixel(Bitmap_t* btmp, uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b);
uint8_t* gexPixel(Bitmap_t* btmp, uint32_t x, uint32_t y, uint8_t* r, uint8_t* g, uint8_t* b);

#endif
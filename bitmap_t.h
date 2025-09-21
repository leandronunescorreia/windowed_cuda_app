#ifndef f BITMAP_T_H
#define BITMAP_T_H

#include <cstdint>

typedef struct {
    uint32_t width;
    uint32_t height;
    uint8_t bitDepth;
    uint8_t* data;
} Bitmap_t;

#endif
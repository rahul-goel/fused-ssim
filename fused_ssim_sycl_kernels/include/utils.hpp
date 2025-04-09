#ifndef FusedSSIMSycl_utils_HPP
#define FusedSSIMSycl_utils_HPP
// ------------------------------------------
// Block and Shared Memory Dimensions
// ------------------------------------------
#define BLOCK_X 16
#define BLOCK_Y 16
#define HALO    5

#define SHARED_X (BLOCK_X + 2 * HALO)
#define SHARED_Y (BLOCK_Y + 2 * HALO)

// For partial results after horizontal pass
#define CONV_X BLOCK_X
#define CONV_Y SHARED_Y

// ------------------------------------------
// Gaussian Coefficients
// ------------------------------------------
constexpr float cGauss[11] = {
    0.001028380123898387f,
    0.0075987582094967365f,
    0.036000773310661316f,
    0.10936068743467331f,
    0.21300552785396576f,
    0.26601171493530273f,
    0.21300552785396576f,
    0.10936068743467331f,
    0.036000773310661316f,
    0.0075987582094967365f,
    0.001028380123898387f
};


// ------------------------------------------
// Utility: Safe pixel fetch w/ zero padding
// ------------------------------------------
inline float get_pix_value(
    const float* img, 
    int b, int c, int y, int x,
    int CH, int H, int W
) {
    if (x < 0 || x >= W || y < 0 || y >= H) {
        return 0.0f;
    }
    return img[b * CH * H * W + c * H * W + y * W + x];
}

#endif //FusedSSIMSycl_utils_HPP
#include "Halide.h"
#include <stdio.h>
#include <string>
#include <cmath>
#include <vector>

using namespace Halide;

// Test configuration: define test sizes (width, height, channels)
struct TestSize {
    int width;
    int height;
    int channels;
    const char* name;
};

const std::vector<TestSize> TEST_SIZES = {
    {1, 1, 3, "1x1x3"},
    {3, 3, 3, "3x3x3"},
    {15, 15, 3, "15x15x3"},
    {16, 16, 3, "16x16x3"},
    {17, 17, 3, "17x17x3"},
    {64, 64, 3, "64x64x3"},
    {128, 128, 3, "128x128x3"},
    {256, 256, 3, "256x256x3"},
    {512, 512, 3, "512x512x3"},
    {4096, 4096, 10, "4096x4096x10"},
};

int main(int argc, char **argv) {
    // Parse target and test flag from command line
    // Usage: ./fused_ssim [target] [--test]
    // target: metal, cuda, opencl, or host (default)
    std::string target_name = "host";
    bool run_tests = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--test" || arg == "-t") {
            run_tests = true;
        } else {
            target_name = arg;
        }
    }

    // Set up target
    Target target = get_host_target();
    if (target_name == "metal") {
        target.set_feature(Target::Metal);
    } else if (target_name == "cuda") {
        target.set_feature(Target::CUDA);
    } else if (target_name == "opencl") {
        target.set_feature(Target::OpenCL);
    }
    // else: use host target (CPU)

    printf("Fused SSIM Halide AOT Generator\n");
    printf("================================\n");
    printf("Usage: ./fused_ssim [target] [--test]\n");
    printf("  target: metal, cuda, opencl, or host (default: host)\n");
    printf("  --test/-t: Run tests on the specified device\n");
    printf("\nTarget: %s\n", target.to_string().c_str());
    printf("Test mode: %s\n", run_tests ? "enabled" : "disabled");
    printf("\n");

    // SSIM Constants (will be passed as runtime parameters)
    Param<float> C1("C1");
    Param<float> C2("C2");

    // Input images as ImageParam (3D: width x height x channels)
    ImageParam img1_param(Float(32), 3, "img1");
    ImageParam img2_param(Float(32), 3, "img2");

    // Variables for indexing
    Var x("x"), y("y"), c("c");
    RDom r(0, 11);  // Reduction domain for 11-tap kernel

    // Gaussian kernel weights (11-tap, sigma=1.5)
    Func gaussian("gaussian");
    gaussian(x) = select(
        x == 0, 0.001028380123898387f,
        x == 1, 0.0075987582094967365f,
        x == 2, 0.036000773310661316f,
        x == 3, 0.10936068743467331f,
        x == 4, 0.21300552785396576f,
        x == 5, 0.26601171493530273f,
        x == 6, 0.21300552785396576f,
        x == 7, 0.10936068743467331f,
        x == 8, 0.036000773310661316f,
        x == 9, 0.0075987582094967365f,
        x == 10, 0.001028380123898387f,
        0.0f
    );

    // Input functions with boundary conditions (constant exterior = 0)
    Func input1 = BoundaryConditions::constant_exterior(img1_param, 0.0f);
    Func input2 = BoundaryConditions::constant_exterior(img2_param, 0.0f);

    // Horizontal pass: compute partial sums for separable convolution
    Func horizontal_pass("horizontal_pass");
    Expr val1 = input1(x + r - 5, y, c);
    Expr val2 = input2(x + r - 5, y, c);
    Expr weight = gaussian(r);

    horizontal_pass(x, y, c) = Tuple(
        sum(weight * val1),            // sum_x (for mu1)
        sum(weight * val1 * val1),     // sum_x2 (for sigma1_sq)
        sum(weight * val2),            // sum_y (for mu2)
        sum(weight * val2 * val2),     // sum_y2 (for sigma2_sq)
        sum(weight * val1 * val2)      // sum_xy (for sigma12)
    );

    // Vertical pass: complete the separable convolution
    // Apply zero padding for out-of-bounds accesses to horizontal_pass
    Expr y_offset = y + r - 5;
    Expr in_bounds = (y_offset >= 0 && y_offset < img1_param.height());

    // Get horizontal_pass values, or use zero if out of bounds
    auto hp = horizontal_pass(x, clamp(y_offset, 0, img1_param.height() - 1), c);
    Expr w0 = select(in_bounds, hp[0], 0.0f);
    Expr w1 = select(in_bounds, hp[1], 0.0f);
    Expr w2 = select(in_bounds, hp[2], 0.0f);
    Expr w3 = select(in_bounds, hp[3], 0.0f);
    Expr w4 = select(in_bounds, hp[4], 0.0f);

    Func vertical_pass("vertical_pass");
    vertical_pass(x, y, c) = Tuple(
        sum(weight * w0),  // mu1
        sum(weight * w1),  // E[X^2]
        sum(weight * w2),  // mu2
        sum(weight * w3),  // E[Y^2]
        sum(weight * w4)   // E[XY]
    );

    // Compute SSIM map and partial derivatives
    Expr mu1 = vertical_pass(x, y, c)[0];
    Expr mu2 = vertical_pass(x, y, c)[2];
    Expr sigma1_sq = vertical_pass(x, y, c)[1] - mu1 * mu1;
    Expr sigma2_sq = vertical_pass(x, y, c)[3] - mu2 * mu2;
    Expr sigma12 = vertical_pass(x, y, c)[4] - mu1 * mu2;

    // SSIM formula components
    Expr A = mu1 * mu1 + mu2 * mu2 + C1;
    Expr B = sigma1_sq + sigma2_sq + C2;
    Expr C_num = 2.0f * mu1 * mu2 + C1;
    Expr D = 2.0f * sigma12 + C2;

    // SSIM map output
    Func ssim_map("ssim_map");
    ssim_map(x, y, c) = (C_num * D) / (A * B);

    // Partial derivatives for backward pass
    Func dm_dmu1("dm_dmu1");
    Func dm_dsigma1_sq("dm_dsigma1_sq");
    Func dm_dsigma12("dm_dsigma12");

    Expr ssim_val = ssim_map(x, y, c);
    dm_dmu1(x, y, c) = (2.0f * D / B) * ((mu2 - mu1 * ssim_val) / A);
    dm_dsigma1_sq(x, y, c) = -ssim_val / B;
    dm_dsigma12(x, y, c) = (2.0f / B) * (1.0f - sigma12 * ssim_val / D);

    // Scheduling
    Var xo("xo"), yo("yo"), xi("xi"), yi("yi");

    // Schedule based on target
    if (target.has_gpu_feature()) {
        printf("Applying GPU schedule...\n");

        // GPU scheduling with 16x16 tiles
        horizontal_pass.compute_root()
                       .gpu_tile(x, y, xo, yo, xi, yi, 16, 16, TailStrategy::GuardWithIf);

        vertical_pass.compute_root()
                     .gpu_tile(x, y, xo, yo, xi, yi, 16, 16, TailStrategy::GuardWithIf);

        ssim_map.gpu_tile(x, y, xo, yo, xi, yi, 16, 16, TailStrategy::GuardWithIf);
        dm_dmu1.gpu_tile(x, y, xo, yo, xi, yi, 16, 16, TailStrategy::GuardWithIf);
        dm_dsigma1_sq.gpu_tile(x, y, xo, yo, xi, yi, 16, 16, TailStrategy::GuardWithIf);
        dm_dsigma12.gpu_tile(x, y, xo, yo, xi, yi, 16, 16, TailStrategy::GuardWithIf);
    } else {
        printf("Applying CPU schedule...\n");

        // CPU scheduling with parallelization and vectorization
        horizontal_pass.compute_root()
                       .parallel(y)
                       .vectorize(x, 8, TailStrategy::GuardWithIf);

        vertical_pass.compute_root()
                     .parallel(y)
                     .vectorize(x, 8, TailStrategy::GuardWithIf);

        ssim_map.parallel(y).vectorize(x, 8, TailStrategy::GuardWithIf);
        dm_dmu1.parallel(y).vectorize(x, 8, TailStrategy::GuardWithIf);
        dm_dsigma1_sq.parallel(y).vectorize(x, 8, TailStrategy::GuardWithIf);
        dm_dsigma12.parallel(y).vectorize(x, 8, TailStrategy::GuardWithIf);
    }

    // Create the pipeline with all outputs
    Pipeline pipeline({ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12});

    // Run tests if requested
    if (run_tests) {
        printf("\n");
        printf("Running tests on target: %s\n", target.to_string().c_str());
        printf("========================================\n");
        printf("Testing %zu different image sizes\n", TEST_SIZES.size());
        printf("========================================\n");

        // Test constants
        const float c1_val = 0.01f;
        const float c2_val = 0.03f;

        int total_tests = 0;
        int passed_tests = 0;

        // Loop through all test sizes
        for (const auto& test_size : TEST_SIZES) {
            const int width = test_size.width;
            const int height = test_size.height;
            const int channels = test_size.channels;

            printf("\n");
            printf("========================================\n");
            printf("Testing size: %s (%d x %d x %d)\n", test_size.name, width, height, channels);
            printf("========================================\n");

            // Create test input buffers
            Buffer<float> test_img1(width, height, channels);
            Buffer<float> test_img2(width, height, channels);

            // Fill with test data: img1 = 0.5, img2 = 0.5 (should give SSIM = 1.0)
            printf("\nTest 1: Identical images (expected SSIM ≈ 1.0)\n");
            for (int c_idx = 0; c_idx < channels; c_idx++) {
                for (int y_idx = 0; y_idx < height; y_idx++) {
                    for (int x_idx = 0; x_idx < width; x_idx++) {
                        test_img1(x_idx, y_idx, c_idx) = 0.5f;
                        test_img2(x_idx, y_idx, c_idx) = 0.5f;
                    }
                }
            }
            test_img1.set_host_dirty();
            test_img2.set_host_dirty();

            // Set parameters
            C1.set(c1_val);
            C2.set(c2_val);
            img1_param.set(test_img1);
            img2_param.set(test_img2);

            // Allocate output buffers
            Buffer<float> out_ssim_map(width, height, channels);
            Buffer<float> out_dm_dmu1(width, height, channels);
            Buffer<float> out_dm_dsigma1_sq(width, height, channels);
            Buffer<float> out_dm_dsigma12(width, height, channels);

            // Realize the pipeline (JIT compile and execute)
            pipeline.realize({out_ssim_map, out_dm_dmu1, out_dm_dsigma1_sq, out_dm_dsigma12}, target);

            // Copy results back from device to host
            out_ssim_map.copy_to_host();
            out_dm_dmu1.copy_to_host();
            out_dm_dsigma1_sq.copy_to_host();
            out_dm_dsigma12.copy_to_host();

            // Verify results - check center pixel of first channel
            float ssim_center = out_ssim_map(width/2, height/2, 0);
            printf("  SSIM at center pixel: %.6f\n", ssim_center);
            printf("  dm_dmu1 at center: %.6f\n", out_dm_dmu1(width/2, height/2, 0));
            printf("  dm_dsigma1_sq at center: %.6f\n", out_dm_dsigma1_sq(width/2, height/2, 0));
            printf("  dm_dsigma12 at center: %.6f\n", out_dm_dsigma12(width/2, height/2, 0));

            // Compute average SSIM
            double ssim_sum = 0.0;
            for (int c_idx = 0; c_idx < channels; c_idx++) {
                for (int y_idx = 0; y_idx < height; y_idx++) {
                    for (int x_idx = 0; x_idx < width; x_idx++) {
                        ssim_sum += out_ssim_map(x_idx, y_idx, c_idx);
                    }
                }
            }
            double ssim_avg = ssim_sum / (width * height * channels);
            printf("  Average SSIM: %.6f\n", ssim_avg);

            bool test1_pass = std::abs(ssim_avg - 1.0) < 0.01;
            printf("  Test 1: %s\n", test1_pass ? "PASS" : "FAIL");
            total_tests++;
            if (test1_pass) passed_tests++;

            // Test 2: Different images
            printf("\nTest 2: Different images (expected SSIM < 1.0)\n");
            for (int c_idx = 0; c_idx < channels; c_idx++) {
                for (int y_idx = 0; y_idx < height; y_idx++) {
                    for (int x_idx = 0; x_idx < width; x_idx++) {
                        test_img1(x_idx, y_idx, c_idx) = 0.3f + 0.4f * (x_idx / float(width));
                        test_img2(x_idx, y_idx, c_idx) = 0.5f + 0.3f * (y_idx / float(height));
                    }
                }
            }
            test_img1.set_host_dirty();
            test_img2.set_host_dirty();

            img1_param.set(test_img1);
            img2_param.set(test_img2);

            pipeline.realize({out_ssim_map, out_dm_dmu1, out_dm_dsigma1_sq, out_dm_dsigma12}, target);

            // Copy results back from device to host
            out_ssim_map.copy_to_host();

            ssim_center = out_ssim_map(width/2, height/2, 0);
            printf("  SSIM at center pixel: %.6f\n", ssim_center);

            ssim_sum = 0.0;
            for (int c_idx = 0; c_idx < channels; c_idx++) {
                for (int y_idx = 0; y_idx < height; y_idx++) {
                    for (int x_idx = 0; x_idx < width; x_idx++) {
                        ssim_sum += out_ssim_map(x_idx, y_idx, c_idx);
                    }
                }
            }
            ssim_avg = ssim_sum / (width * height * channels);
            printf("  Average SSIM: %.6f\n", ssim_avg);

            bool test2_pass = ssim_avg < 1.0 && ssim_avg > 0.0;
            printf("  Test 2: %s\n", test2_pass ? "PASS" : "FAIL");
            total_tests++;
            if (test2_pass) passed_tests++;

            // Size summary
            if (test1_pass && test2_pass) {
                printf("  %s: ALL TESTS PASSED\n", test_size.name);
            } else {
                printf("  %s: SOME TESTS FAILED\n", test_size.name);
            }
        } // End of test size loop

        // Overall Summary
        printf("\n========================================\n");
        printf("OVERALL TEST SUMMARY\n");
        printf("========================================\n");
        printf("Target: %s\n", target.to_string().c_str());
        printf("Tests passed: %d / %d\n", passed_tests, total_tests);
        printf("Test sizes: %zu\n", TEST_SIZES.size());
        if (passed_tests == total_tests) {
            printf("\n✓ ALL TESTS PASSED\n");
        } else {
            printf("\n✗ SOME TESTS FAILED\n");
        }
        printf("========================================\n\n");
    }

    // Compile to static library
    std::string output_name = "ssim_halide";
    printf("Compiling to static library: %s.a, %s.h\n", output_name.c_str(), output_name.c_str());

    pipeline.compile_to_static_library(
        output_name,
        {img1_param, img2_param, C1, C2},  // Inputs: images + constants
        "fused_ssim_forward",               // Function name
        target
    );

    printf("\nSUCCESS: Generated %s.a and %s.h\n", output_name.c_str(), output_name.c_str());

    printf("\nGenerated function signature:\n");
    printf("  int fused_ssim_forward(\n");
    printf("      halide_buffer_t *img1,\n");
    printf("      halide_buffer_t *img2,\n");
    printf("      float C1,\n");
    printf("      float C2,\n");
    printf("      halide_buffer_t *ssim_map,\n");
    printf("      halide_buffer_t *dm_dmu1,\n");
    printf("      halide_buffer_t *dm_dsigma1_sq,\n");
    printf("      halide_buffer_t *dm_dsigma12\n");
    printf("  );\n");

    return 0;
}

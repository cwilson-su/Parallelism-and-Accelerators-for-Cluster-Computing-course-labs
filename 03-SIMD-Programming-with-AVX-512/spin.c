/*
Wilson Cedric Genevieve
LAB3 PACC Course
Nov 26
*/

#include "easypap.h"

#include <math.h>
#include <omp.h>

#define M_PIsp (float)M_PI
#define M_PI_2sp (float)M_PI_2

static void rotate (void);
static unsigned compute_color (int i, int j);

// If defined, the initialization hook function is called quite early in the
// initialization process, after the size (DIM variable) of images is known.
// This function can typically spawn a team of threads, or allocated additionnal
// OpenCL buffers.
// A function named <kernel>_init_<variant> is search first. If not found, a
// function <kernel>_init is searched in turn.
void spin_init (void)
{
  // check tile size's conformity with respect to CPU vector width
  // easypap_check_vectorization (VEC_TYPE_FLOAT, DIR_HORIZONTAL);

  PRINT_DEBUG ('u', "Image size is %dx%d\n", DIM, DIM);
  PRINT_DEBUG ('u', "Block size is %dx%d\n", TILE_W, TILE_H);
  PRINT_DEBUG ('u', "Press <SPACE> to pause/unpause, <ESC> to quit.\n");
}

// The image is a two-dimension array of size of DIM x DIM. Each pixel is of
// type 'unsigned' and store the color information following a RGBA layout (4
// bytes). Pixel at line 'l' and column 'c' in the current image can be accessed
// using cur_img (l, c).

// The kernel returns 0, or the iteration step at which computation has
// completed (e.g. stabilized).

///////////////////////////// Simple sequential version (seq)
// Suggested cmdline(s):
// ./run --size 1024 --kernel spin --variant seq
// or
// ./run -s 1024 -k spin -v seq
//
unsigned spin_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int i = 0; i < DIM; i++)
      for (int j = 0; j < DIM; j++)
        cur_img (i, j) = compute_color (i, j);

    rotate (); // Slightly increase the base angle
  }

  return 0;
}





// Tile computation
int spin_do_tile_default (int x, int y, int width, int height)
{
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      cur_img (i, j) = compute_color (i, j);

  return 0;
}


///////////////////////////// Tiled sequential version (tiled)
// Suggested cmdline(s):
// ./run -k spin -v tiled -ts 64 -m
//
unsigned spin_compute_tiled (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        do_tile (x, y, TILE_W, TILE_H);

    rotate ();
  }

  return 0;
}


//////////////////////////////////////////////////////////////////////////

static float base_angle = 0.0f;
static float yellow_r = 255.0f, yellow_g = 255.0f, yellow_b = 0.0f,
             yellow_a = 255.0f;
static float blue_r = 0.0f, blue_g = 0.0f, blue_b = 255.0f, blue_a = 255.0f;

// Computation of one pixel
static unsigned compute_color (int i, int j)
{
  float atan2f_in1 = (float)DIM / 2.f - (float)i;
  float atan2f_in2 = (float)j - (float)DIM / 2.f;
  float angle = atan2f(atan2f_in1, atan2f_in2) + M_PIsp + base_angle;

  float ratio = fabsf((float)((fmodf(angle, M_PIsp / 4.f) - (M_PIsp / 8.f)) /
    (M_PIsp / 8.f)));

  uint32_t r = blue_r + (yellow_r - blue_r) * ratio;
  uint32_t g = blue_g + (yellow_g - blue_g) * ratio;
  uint32_t b = blue_b + (yellow_b - blue_b) * ratio;
  uint32_t a = blue_a + (yellow_a - blue_a) * ratio;

  return ezv_rgba(r, g, b, a);
}



static void rotate (void)
{
  base_angle = fmodf (base_angle + (1.f / 180.f) * M_PIsp, M_PIsp);
}



// ----------------------------------------------------------------------------
// --------------------------------------------------------- APPROX SEQ VERSION
// ----------------------------------------------------------------------------

// ####################LAB Q 2.1################
static float atanf_approx (float x) {
  return x * M_PIsp / 4.f + 0.273f * x * (1.f - fabsf(x));
}

static float atan2f_approx (float y, float x) {
  float ay = fabsf(y);
  float ax = fabsf(x);
  int invert = ay > ax;
  float z = invert ? ax / ay : ay / ax; 
  float th = atanf_approx(z); 
  if (invert) th = M_PI_2sp - th; 
  if (x < 0) th = M_PIsp - th; 
  if (y < 0) th = -th;
  return th;
}

static float fmodf_approx (float x, float y) {
  return x - trunc(x / y) * y;
}

// Computation of one pixel
static unsigned compute_color_approx (int i, int j)
{
  float atan2f_in1 = (float)DIM / 2.f - (float)i;
  float atan2f_in2 = (float)j - (float)DIM / 2.f;
  float angle = atan2f_approx (atan2f_in1, atan2f_in2) + M_PIsp + base_angle;

  float ratio = fabsf ((fmodf_approx (angle, M_PIsp / 4.f) - (M_PIsp / 8.f)) / (M_PIsp / 8.f));

  uint32_t r = blue_r + (yellow_r - blue_r) * ratio;
  uint32_t g = blue_g + (yellow_g - blue_g) * ratio;
  uint32_t b = blue_b + (yellow_b - blue_b) * ratio;
  uint32_t a = blue_a + (yellow_a - blue_a) * ratio;

  return ezv_rgba(r, g, b, a);
}


unsigned spin_compute_approx (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int i = 0; i < DIM; i++)
      for (int j = 0; j < DIM; j++)
        cur_img (i, j) = compute_color_approx (i, j);

    rotate (); // Slightly increase the base angle
  }

  return 0;
}


#ifdef ENABLE_VECTO

#if defined(__AVX512F__)

#include <immintrin.h>
// ----------------------------------------------------------------------------
// ------------------------------------------------------------- SIMD VERSION 0
// ----------------------------------------------------------------------------

// Computation of one pixel
static inline __m512i compute_color_simd_v0(__m512i r_i, __m512i r_j) {
    int32_t temp_i[16];
    int32_t temp_j[16];
    int32_t temp_out[16];

    // Safely dump vector registers to memory
    _mm512_storeu_epi32(temp_i, r_i);
    _mm512_storeu_epi32(temp_j, r_j);

    for (int k = 0; k < 16; k++) {
        temp_out[k] = compute_color_approx(temp_i[k], temp_j[k]);
    }

    return _mm512_loadu_epi32(temp_out);
}

unsigned spin_compute_simd_v0(unsigned nb_iter) {
  int32_t tab_j[16];
  for (unsigned it = 1; it <= nb_iter; it++) {
    for (unsigned i = 0; i < DIM; i++)
      for (unsigned j = 0; j < DIM; j += 16) {
        for (unsigned jj = 0; jj < 16; jj++) tab_j[jj] = j + jj;
        int* img_out_ptr = (int*)&cur_img(i, j);
        __m512i r_i = _mm512_set1_epi32(i);
        __m512i r_j = _mm512_loadu_epi32(tab_j);
        __m512i r_result = compute_color_simd_v0(r_i, r_j);
        _mm512_storeu_epi32(img_out_ptr, r_result);
      }

    rotate();  // Slightly increase the base angle
  }

  return 0;
}

// ----------------------------------------------------------------------------
// ------------------------------------------------------------- SIMD VERSION 1
// ----------------------------------------------------------------------------

// 2.3.1
static inline __m512 fmodf_approx_simd(__m512 r_x, __m512 r_y) {
    __m512 r_div = _mm512_div_ps(r_x, r_y);
    __m512 r_trunc = _mm512_roundscale_ps(r_div, _MM_FROUND_TO_ZERO);
    __m512 r_mul = _mm512_mul_ps(r_trunc, r_y);
    return _mm512_sub_ps(r_x, r_mul);
}

// 2.3.2 Computation of one pixel
static inline __m512i compute_color_simd_v1(__m512i r_i, __m512i r_j) {
    int32_t temp_i[16];
    int32_t temp_j[16];
    
    // Safely dump vector registers to memory
    _mm512_storeu_epi32(temp_i, r_i);
    _mm512_storeu_epi32(temp_j, r_j);
    
    float temp_angles[16];
    float temp_ratios[16];
    int32_t temp_out[16];

    // 1. LOOP 1: Sequential math up to 'angle'
    for (int k = 0; k < 16; k++) {
        float atan2f_in1 = (float)DIM / 2.f - (float)temp_i[k];
        float atan2f_in2 = (float)temp_j[k] - (float)DIM / 2.f;
        temp_angles[k] = atan2f_approx(atan2f_in1, atan2f_in2) + M_PIsp + base_angle;
    }

    // 2. SIMD MATH: fmodf and ratio (all 16 at once!)
    __m512 r_angle = _mm512_loadu_ps(temp_angles);
    __m512 r_pi_4  = _mm512_set1_ps(M_PIsp / 4.f);
    __m512 r_pi_8  = _mm512_set1_ps(M_PIsp / 8.f);

    __m512 r_fmod  = fmodf_approx_simd(r_angle, r_pi_4);
    __m512 r_sub   = _mm512_sub_ps(r_fmod, r_pi_8);
    __m512 r_div   = _mm512_div_ps(r_sub, r_pi_8);
    __m512 r_ratio = _mm512_abs_ps(r_div); // SIMD version of fabsf

    _mm512_storeu_ps(temp_ratios, r_ratio); // Save ratios back to memory

    // 3. LOOP 2: Sequential color packing
    for (int k = 0; k < 16; k++) {
        float ratio = temp_ratios[k];
        uint32_t r = blue_r + (yellow_r - blue_r) * ratio;
        uint32_t g = blue_g + (yellow_g - blue_g) * ratio;
        uint32_t b = blue_b + (yellow_b - blue_b) * ratio;
        uint32_t a = blue_a + (yellow_a - blue_a) * ratio;
        temp_out[k] = ezv_rgba(r, g, b, a);
    }

    return _mm512_loadu_epi32(temp_out);
}

unsigned spin_compute_simd_v1(unsigned nb_iter) {
  int32_t tab_j[16];
  for (unsigned it = 1; it <= nb_iter; it++) {
    for (unsigned i = 0; i < DIM; i++)
      for (unsigned j = 0; j < DIM; j += 16) {
        for (unsigned jj = 0; jj < 16; jj++) tab_j[jj] = j + jj;
        int* img_out_ptr = (int*)&cur_img(i, j);
        __m512i r_i = _mm512_set1_epi32(i);
        __m512i r_j = _mm512_loadu_epi32(tab_j);
        __m512i r_result = compute_color_simd_v1(r_i, r_j);
        _mm512_storeu_epi32(img_out_ptr, r_result);
      }
    rotate();  // Slightly increase the base angle
  }

  return 0;
}


// ----------------------------------------------------------------------------
// ------------------------------------------------------------- SIMD VERSION 2
// ----------------------------------------------------------------------------

// 2.4.1
static inline __m512i rgba_simd(__m512i r_r, __m512i r_g, __m512i r_b, __m512i r_a) {
    __m512i shifted_g = _mm512_slli_epi32(r_g, 8);
    __m512i shifted_b = _mm512_slli_epi32(r_b, 16);
    __m512i shifted_a = _mm512_slli_epi32(r_a, 24);

    __m512i out = _mm512_or_epi32(r_r, shifted_g);
    out = _mm512_or_epi32(out, shifted_b);
    out = _mm512_or_epi32(out, shifted_a);

    return out;
}

// 2.4.2 - Computation of one pixel
static inline __m512i compute_color_simd_v2(__m512i r_i, __m512i r_j) {
    int32_t temp_i[16];
    int32_t temp_j[16];
    _mm512_storeu_epi32(temp_i, r_i);
    _mm512_storeu_epi32(temp_j, r_j);
    
    float temp_angles[16];

    // 1. LOOP 1: Sequential math up to 'angle'
    for (int k = 0; k < 16; k++) {
        float atan2f_in1 = (float)DIM / 2.f - (float)temp_i[k];
        float atan2f_in2 = (float)temp_j[k] - (float)DIM / 2.f;
        temp_angles[k] = atan2f_approx(atan2f_in1, atan2f_in2) + M_PIsp + base_angle;
    }

    // 2. SIMD MATH: fmodf and ratio (all 16 at once)
    __m512 r_angle = _mm512_loadu_ps(temp_angles);
    __m512 r_pi_4  = _mm512_set1_ps(M_PIsp / 4.f);
    __m512 r_pi_8  = _mm512_set1_ps(M_PIsp / 8.f);

    __m512 r_fmod  = fmodf_approx_simd(r_angle, r_pi_4);
    __m512 r_sub   = _mm512_sub_ps(r_fmod, r_pi_8);
    __m512 r_div   = _mm512_div_ps(r_sub, r_pi_8);
    __m512 r_ratio = _mm512_abs_ps(r_div);

    // 3. SIMD MATH: Calculate final colors (blue + (diff * ratio))
    __m512 r_blue_r = _mm512_set1_ps((float)blue_r);
    __m512 r_blue_g = _mm512_set1_ps((float)blue_g);
    __m512 r_blue_b = _mm512_set1_ps((float)blue_b);
    __m512 r_blue_a = _mm512_set1_ps((float)blue_a);

    __m512 r_diff_r = _mm512_set1_ps((float)(yellow_r - blue_r));
    __m512 r_diff_g = _mm512_set1_ps((float)(yellow_g - blue_g));
    __m512 r_diff_b = _mm512_set1_ps((float)(yellow_b - blue_b));
    __m512 r_diff_a = _mm512_set1_ps((float)(yellow_a - blue_a));

    __m512 r_r_float = _mm512_add_ps(r_blue_r, _mm512_mul_ps(r_diff_r, r_ratio));
    __m512 r_g_float = _mm512_add_ps(r_blue_g, _mm512_mul_ps(r_diff_g, r_ratio));
    __m512 r_b_float = _mm512_add_ps(r_blue_b, _mm512_mul_ps(r_diff_b, r_ratio));
    __m512 r_a_float = _mm512_add_ps(r_blue_a, _mm512_mul_ps(r_diff_a, r_ratio));

    // Convert floating point colors back to integers
    __m512i r_r_int = _mm512_cvtps_epi32(r_r_float);
    __m512i r_g_int = _mm512_cvtps_epi32(r_g_float);
    __m512i r_b_int = _mm512_cvtps_epi32(r_b_float);
    __m512i r_a_int = _mm512_cvtps_epi32(r_a_float);

    // Pack them into 32-bit pixels in parallel
    return rgba_simd(r_r_int, r_g_int, r_b_int, r_a_int);
}


unsigned spin_compute_simd_v2(unsigned nb_iter) {
  int32_t tab_j[16];
  for (unsigned it = 1; it <= nb_iter; it++) {
    for (unsigned i = 0; i < DIM; i++)
      for (unsigned j = 0; j < DIM; j += 16) {
        for (unsigned jj = 0; jj < 16; jj++) tab_j[jj] = j + jj;
        int* img_out_ptr = (int*)&cur_img(i, j);
        __m512i r_i = _mm512_set1_epi32(i);
        __m512i r_j = _mm512_loadu_epi32(tab_j);
        __m512i r_result = compute_color_simd_v2(r_i, r_j);
        _mm512_storeu_epi32(img_out_ptr, r_result);
      }
    rotate();  // Slightly increase the base angle
  }

  return 0;
}

// ----------------------------------------------------------------------------
// ------------------------------------------------------------- SIMD VERSION 3
// ----------------------------------------------------------------------------

static inline __m512 atanf_approx_simd(__m512 r_x) {
    __m512 r_1 = _mm512_set1_ps(1.0f);
    __m512 r_028 = _mm512_set1_ps(0.28f);
    __m512 r_pi_2 = _mm512_set1_ps(M_PIsp / 2.0f);

    __m512 r_x2 = _mm512_mul_ps(r_x, r_x);

    // Path 1 (True): M_PIsp / 2.f - x / (x * x + 0.28f)
    __m512 r_den1 = _mm512_add_ps(r_x2, r_028);
    __m512 r_div1 = _mm512_div_ps(r_x, r_den1);
    __m512 r_path1 = _mm512_sub_ps(r_pi_2, r_div1);

    // Path 2 (False): x / (1.f + 0.28f * x * x)
    __m512 r_den2 = _mm512_add_ps(r_1, _mm512_mul_ps(r_028, r_x2));
    __m512 r_path2 = _mm512_div_ps(r_x, r_den2);

    // Mask and Blend! (Note: Intel blends take the 'false' path first, then the 'true' path)
    __mmask16 mask_gt = _mm512_cmp_ps_mask(r_x, r_1, _CMP_GT_OQ);
    return _mm512_mask_blend_ps(mask_gt, r_path2, r_path1);
}

static inline __m512 atan2f_approx_simd(__m512 r_y, __m512 r_x) {
    __m512 r_ay = _mm512_abs_ps(r_y);
    __m512 r_ax = _mm512_abs_ps(r_x);

    // int invert = ay > ax;
    __mmask16 mask_invert = _mm512_cmp_ps_mask(r_ay, r_ax, _CMP_GT_OQ);

    // float z = invert ? ax / ay : ay / ax;
    __m512 r_div_inv = _mm512_div_ps(r_ax, r_ay);
    __m512 r_div_not_inv = _mm512_div_ps(r_ay, r_ax);
    __m512 r_z = _mm512_mask_blend_ps(mask_invert, r_div_not_inv, r_div_inv);

    // float th = atanf_approx(z);
    __m512 r_th = atanf_approx_simd(r_z);

    // if (invert) th = M_PI_2sp - th;
    __m512 r_pi_2 = _mm512_set1_ps(M_PI_2sp);
    __m512 r_th_inv = _mm512_sub_ps(r_pi_2, r_th);
    r_th = _mm512_mask_blend_ps(mask_invert, r_th, r_th_inv);

    // if (x < 0) th = M_PIsp - th;
    __m512 r_zero = _mm512_setzero_ps();
    __mmask16 mask_x_lt_0 = _mm512_cmp_ps_mask(r_x, r_zero, _CMP_LT_OQ);
    __m512 r_pi = _mm512_set1_ps(M_PIsp);
    __m512 r_th_x_lt_0 = _mm512_sub_ps(r_pi, r_th);
    r_th = _mm512_mask_blend_ps(mask_x_lt_0, r_th, r_th_x_lt_0);

    // if (y < 0) th = -th;
    __mmask16 mask_y_lt_0 = _mm512_cmp_ps_mask(r_y, r_zero, _CMP_LT_OQ);
    __m512 r_th_neg = _mm512_sub_ps(r_zero, r_th);
    r_th = _mm512_mask_blend_ps(mask_y_lt_0, r_th, r_th_neg);

    return r_th;
}


// Computation of one pixel
static inline __m512i compute_color_simd_v3(__m512i r_i, __m512i r_j) {
    // Convert int to float
    __m512 r_float_i = _mm512_cvtepi32_ps(r_i);
    __m512 r_float_j = _mm512_cvtepi32_ps(r_j);
    __m512 r_dim_2 = _mm512_set1_ps((float)DIM / 2.f);

    // atan2f_in1 and in2
    __m512 r_atan2f_in1 = _mm512_sub_ps(r_dim_2, r_float_i);
    __m512 r_atan2f_in2 = _mm512_sub_ps(r_float_j, r_dim_2);

    // atan2f_approx_simd + M_PIsp + base_angle
    __m512 r_th = atan2f_approx_simd(r_atan2f_in1, r_atan2f_in2);
    __m512 r_pi_base = _mm512_set1_ps(M_PIsp + base_angle);
    __m512 r_angle = _mm512_add_ps(r_th, r_pi_base);

    // fmodf and ratio
    __m512 r_pi_4 = _mm512_set1_ps(M_PIsp / 4.f);
    __m512 r_pi_8 = _mm512_set1_ps(M_PIsp / 8.f);
    __m512 r_fmod = fmodf_approx_simd(r_angle, r_pi_4);
    __m512 r_sub = _mm512_sub_ps(r_fmod, r_pi_8);
    __m512 r_div = _mm512_div_ps(r_sub, r_pi_8);
    __m512 r_ratio = _mm512_abs_ps(r_div);

    // Final colors
    __m512 r_blue_r = _mm512_set1_ps((float)blue_r);
    __m512 r_blue_g = _mm512_set1_ps((float)blue_g);
    __m512 r_blue_b = _mm512_set1_ps((float)blue_b);
    __m512 r_blue_a = _mm512_set1_ps((float)blue_a);

    __m512 r_diff_r = _mm512_set1_ps((float)(yellow_r - blue_r));
    __m512 r_diff_g = _mm512_set1_ps((float)(yellow_g - blue_g));
    __m512 r_diff_b = _mm512_set1_ps((float)(yellow_b - blue_b));
    __m512 r_diff_a = _mm512_set1_ps((float)(yellow_a - blue_a));

    __m512 r_r_float = _mm512_add_ps(r_blue_r, _mm512_mul_ps(r_diff_r, r_ratio));
    __m512 r_g_float = _mm512_add_ps(r_blue_g, _mm512_mul_ps(r_diff_g, r_ratio));
    __m512 r_b_float = _mm512_add_ps(r_blue_b, _mm512_mul_ps(r_diff_b, r_ratio));
    __m512 r_a_float = _mm512_add_ps(r_blue_a, _mm512_mul_ps(r_diff_a, r_ratio));

    // Convert back to integers and pack
    return rgba_simd(
        _mm512_cvtps_epi32(r_r_float),
        _mm512_cvtps_epi32(r_g_float),
        _mm512_cvtps_epi32(r_b_float),
        _mm512_cvtps_epi32(r_a_float)
    );
}


unsigned spin_compute_simd_v3(unsigned nb_iter) {
  int32_t tab_j[16];
  for (unsigned it = 1; it <= nb_iter; it++) {
    for (unsigned i = 0; i < DIM; i++)
      for (unsigned j = 0; j < DIM; j += 16) {
        for (unsigned jj = 0; jj < 16; jj++) tab_j[jj] = j + jj;
        int* img_out_ptr = (int*)&cur_img(i, j);
        __m512i r_i = _mm512_set1_epi32(i);
        __m512i r_j = _mm512_loadu_epi32(tab_j);
        __m512i r_result = compute_color_simd_v3(r_i, r_j);
        _mm512_storeu_epi32(img_out_ptr, r_result);
      }
    rotate();  // Slightly increase the base angle
  }

  return 0;
}

// ----------------------------------------------------------------------------
// ------------------------------------------------------------- SIMD VERSION 4
// ----------------------------------------------------------------------------

// Computation of one pixel
static inline __m512i compute_color_simd_v4_task1(__m512i r_i, __m512i r_j) {
    // 1. Declare ALL constants exactly once at the top!
    __m512 r_dim_2   = _mm512_set1_ps((float)DIM / 2.f);
    __m512 r_1       = _mm512_set1_ps(1.0f);
    __m512 r_028     = _mm512_set1_ps(0.28f);
    __m512 r_pi_2    = _mm512_set1_ps(M_PI_2sp);
    __m512 r_pi      = _mm512_set1_ps(M_PIsp);
    __m512 r_zero    = _mm512_setzero_ps();
    __m512 r_pi_base = _mm512_set1_ps(M_PIsp + base_angle);
    __m512 r_pi_4    = _mm512_set1_ps(M_PIsp / 4.f);
    __m512 r_pi_8    = _mm512_set1_ps(M_PIsp / 8.f);

    __m512 r_blue_r = _mm512_set1_ps((float)blue_r);
    __m512 r_blue_g = _mm512_set1_ps((float)blue_g);
    __m512 r_blue_b = _mm512_set1_ps((float)blue_b);
    __m512 r_blue_a = _mm512_set1_ps((float)blue_a);
    __m512 r_diff_r = _mm512_set1_ps((float)(yellow_r - blue_r));
    __m512 r_diff_g = _mm512_set1_ps((float)(yellow_g - blue_g));
    __m512 r_diff_b = _mm512_set1_ps((float)(yellow_b - blue_b));
    __m512 r_diff_a = _mm512_set1_ps((float)(yellow_a - blue_a));

    // 2. Convert ints to floats and set up inputs
    __m512 r_y = _mm512_sub_ps(r_dim_2, _mm512_cvtepi32_ps(r_i));
    __m512 r_x = _mm512_sub_ps(_mm512_cvtepi32_ps(r_j), r_dim_2);

    // 3. --- INLINED atan2f and atanf math ---
    __m512 r_ay = _mm512_abs_ps(r_y);
    __m512 r_ax = _mm512_abs_ps(r_x);
    __mmask16 mask_invert = _mm512_cmp_ps_mask(r_ay, r_ax, _CMP_GT_OQ);
    
    __m512 r_z = _mm512_mask_blend_ps(mask_invert, 
                                      _mm512_div_ps(r_ay, r_ax), 
                                      _mm512_div_ps(r_ax, r_ay));

    __m512 r_z2 = _mm512_mul_ps(r_z, r_z);
    __m512 r_path1 = _mm512_sub_ps(r_pi_2, _mm512_div_ps(r_z, _mm512_add_ps(r_z2, r_028)));
    __m512 r_path2 = _mm512_div_ps(r_z, _mm512_add_ps(r_1, _mm512_mul_ps(r_028, r_z2)));

    __mmask16 mask_gt = _mm512_cmp_ps_mask(r_z, r_1, _CMP_GT_OQ);
    __m512 r_th = _mm512_mask_blend_ps(mask_gt, r_path2, r_path1);

    r_th = _mm512_mask_blend_ps(mask_invert, r_th, _mm512_sub_ps(r_pi_2, r_th));
    r_th = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(r_x, r_zero, _CMP_LT_OQ), r_th, _mm512_sub_ps(r_pi, r_th));
    r_th = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(r_y, r_zero, _CMP_LT_OQ), r_th, _mm512_sub_ps(r_zero, r_th));

    // 4. --- INLINED fmodf and ratio ---
    __m512 r_angle = _mm512_add_ps(r_th, r_pi_base);
    
    __m512 r_div_fmod = _mm512_div_ps(r_angle, r_pi_4);
    __m512 r_trunc = _mm512_roundscale_ps(r_div_fmod, _MM_FROUND_TO_ZERO);
    __m512 r_fmod = _mm512_sub_ps(r_angle, _mm512_mul_ps(r_trunc, r_pi_4));

    __m512 r_ratio = _mm512_abs_ps(_mm512_div_ps(_mm512_sub_ps(r_fmod, r_pi_8), r_pi_8));

    // 5. Final Colors
    __m512 r_r_float = _mm512_add_ps(r_blue_r, _mm512_mul_ps(r_diff_r, r_ratio));
    __m512 r_g_float = _mm512_add_ps(r_blue_g, _mm512_mul_ps(r_diff_g, r_ratio));
    __m512 r_b_float = _mm512_add_ps(r_blue_b, _mm512_mul_ps(r_diff_b, r_ratio));
    __m512 r_a_float = _mm512_add_ps(r_blue_a, _mm512_mul_ps(r_diff_a, r_ratio));

    return rgba_simd(
        _mm512_cvtps_epi32(r_r_float),
        _mm512_cvtps_epi32(r_g_float),
        _mm512_cvtps_epi32(r_b_float),
        _mm512_cvtps_epi32(r_a_float)
    );
}

static inline __m512i compute_color_simd_v4(__m512i r_i, __m512i r_j) {
    // 1. Constants (declared once!)
    __m512 r_dim_2   = _mm512_set1_ps((float)DIM / 2.f);
    __m512 r_1       = _mm512_set1_ps(1.0f);
    __m512 r_028     = _mm512_set1_ps(0.28f);
    __m512 r_pi_2    = _mm512_set1_ps(M_PI_2sp);
    __m512 r_pi      = _mm512_set1_ps(M_PIsp);
    __m512 r_zero    = _mm512_setzero_ps();
    __m512 r_pi_base = _mm512_set1_ps(M_PIsp + base_angle);
    __m512 r_pi_4    = _mm512_set1_ps(M_PIsp / 4.f);
    __m512 r_pi_8    = _mm512_set1_ps(M_PIsp / 8.f);

    __m512 r_blue_r = _mm512_set1_ps((float)blue_r);
    __m512 r_blue_g = _mm512_set1_ps((float)blue_g);
    __m512 r_blue_b = _mm512_set1_ps((float)blue_b);
    __m512 r_blue_a = _mm512_set1_ps((float)blue_a);
    __m512 r_diff_r = _mm512_set1_ps((float)(yellow_r - blue_r));
    __m512 r_diff_g = _mm512_set1_ps((float)(yellow_g - blue_g));
    __m512 r_diff_b = _mm512_set1_ps((float)(yellow_b - blue_b));
    __m512 r_diff_a = _mm512_set1_ps((float)(yellow_a - blue_a));

    // 2. Setup inputs
    __m512 r_y = _mm512_sub_ps(r_dim_2, _mm512_cvtepi32_ps(r_i));
    __m512 r_x = _mm512_sub_ps(_mm512_cvtepi32_ps(r_j), r_dim_2);

    // 3. INLINED atan2f and atanf
    __m512 r_ay = _mm512_abs_ps(r_y);
    __m512 r_ax = _mm512_abs_ps(r_x);
    __mmask16 mask_invert = _mm512_cmp_ps_mask(r_ay, r_ax, _CMP_GT_OQ);
    
    __m512 r_z = _mm512_mask_blend_ps(mask_invert, 
                                      _mm512_div_ps(r_ay, r_ax), 
                                      _mm512_div_ps(r_ax, r_ay));

    __m512 r_z2 = _mm512_mul_ps(r_z, r_z);
    
    __m512 r_den1 = _mm512_add_ps(r_z2, r_028); 
    __m512 r_path1 = _mm512_sub_ps(r_pi_2, _mm512_div_ps(r_z, r_den1));
    
    // FMA optimization: (0.28 * z2) + 1.0
    __m512 r_den2 = _mm512_fmadd_ps(r_028, r_z2, r_1);
    __m512 r_path2 = _mm512_div_ps(r_z, r_den2);

    __mmask16 mask_gt = _mm512_cmp_ps_mask(r_z, r_1, _CMP_GT_OQ);
    __m512 r_th = _mm512_mask_blend_ps(mask_gt, r_path2, r_path1);

    r_th = _mm512_mask_blend_ps(mask_invert, r_th, _mm512_sub_ps(r_pi_2, r_th));
    r_th = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(r_x, r_zero, _CMP_LT_OQ), r_th, _mm512_sub_ps(r_pi, r_th));
    r_th = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(r_y, r_zero, _CMP_LT_OQ), r_th, _mm512_sub_ps(r_zero, r_th));

    // 4. INLINED fmodf and ratio
    __m512 r_angle = _mm512_add_ps(r_th, r_pi_base);
    
    __m512 r_div_fmod = _mm512_div_ps(r_angle, r_pi_4);
    __m512 r_trunc = _mm512_roundscale_ps(r_div_fmod, _MM_FROUND_TO_ZERO);
    
    // FNMA optimization: r_angle - (r_trunc * r_pi_4)
    __m512 r_fmod = _mm512_fnmadd_ps(r_trunc, r_pi_4, r_angle);

    __m512 r_ratio = _mm512_abs_ps(_mm512_div_ps(_mm512_sub_ps(r_fmod, r_pi_8), r_pi_8));

    // 5. Final Colors using FMA! (diff * ratio) + blue
    __m512 r_r_float = _mm512_fmadd_ps(r_diff_r, r_ratio, r_blue_r);
    __m512 r_g_float = _mm512_fmadd_ps(r_diff_g, r_ratio, r_blue_g);
    __m512 r_b_float = _mm512_fmadd_ps(r_diff_b, r_ratio, r_blue_b);
    __m512 r_a_float = _mm512_fmadd_ps(r_diff_a, r_ratio, r_blue_a);

    return rgba_simd(
        _mm512_cvtps_epi32(r_r_float),
        _mm512_cvtps_epi32(r_g_float),
        _mm512_cvtps_epi32(r_b_float),
        _mm512_cvtps_epi32(r_a_float)
    );
}


unsigned spin_compute_simd_v4(unsigned nb_iter) {
  int32_t tab_j[16];
  for (unsigned it = 1; it <= nb_iter; it++) {
    for (unsigned i = 0; i < DIM; i++)
      for (unsigned j = 0; j < DIM; j += 16) {
        for (unsigned jj = 0; jj < 16; jj++) tab_j[jj] = j + jj;
        int* img_out_ptr = (int*)&cur_img(i, j);
        __m512i r_i = _mm512_set1_epi32(i);
        __m512i r_j = _mm512_loadu_epi32(tab_j);
        __m512i r_result = compute_color_simd_v4(r_i, r_j);
        _mm512_storeu_epi32(img_out_ptr, r_result);
      }
    rotate();  // Slightly increase the base angle
  }

  return 0;
}

// ----------------------------------------------------------------------------
// ------------------------------------------------------------- SIMD VERSION 5
// ----------------------------------------------------------------------------

unsigned spin_compute_simd_v5(unsigned nb_iter) {
    int32_t tab_j[16];
    
    for (unsigned it = 1; it <= nb_iter; it++) {
        // --- POSITION A: Constants (Only initialized once per frame!) ---
        __m512 r_dim_2   = _mm512_set1_ps((float)DIM / 2.f);
        __m512 r_1       = _mm512_set1_ps(1.0f);
        __m512 r_028     = _mm512_set1_ps(0.28f);
        __m512 r_pi_2    = _mm512_set1_ps(M_PI_2sp);
        __m512 r_pi      = _mm512_set1_ps(M_PIsp);
        __m512 r_zero    = _mm512_setzero_ps();
        __m512 r_pi_base = _mm512_set1_ps(M_PIsp + base_angle);
        __m512 r_pi_4    = _mm512_set1_ps(M_PIsp / 4.f);
        __m512 r_pi_8    = _mm512_set1_ps(M_PIsp / 8.f);

        __m512 r_blue_r = _mm512_set1_ps((float)blue_r);
        __m512 r_blue_g = _mm512_set1_ps((float)blue_g);
        __m512 r_blue_b = _mm512_set1_ps((float)blue_b);
        __m512 r_blue_a = _mm512_set1_ps((float)blue_a);
        __m512 r_diff_r = _mm512_set1_ps((float)(yellow_r - blue_r));
        __m512 r_diff_g = _mm512_set1_ps((float)(yellow_g - blue_g));
        __m512 r_diff_b = _mm512_set1_ps((float)(yellow_b - blue_b));
        __m512 r_diff_a = _mm512_set1_ps((float)(yellow_a - blue_a));

        for (unsigned i = 0; i < DIM; i++) {
            // --- POSITION B: Row-dependent math (Calculated once per row!) ---
            __m512i r_i = _mm512_set1_epi32(i);
            __m512 r_y = _mm512_sub_ps(r_dim_2, _mm512_cvtepi32_ps(r_i));
            __m512 r_ay = _mm512_abs_ps(r_y);
            __mmask16 mask_y_lt_0 = _mm512_cmp_ps_mask(r_y, r_zero, _CMP_LT_OQ);

            for (unsigned j = 0; j < DIM; j += 16) {
                // --- POSITION C: Column-dependent math ---
                for (unsigned jj = 0; jj < 16; jj++) tab_j[jj] = j + jj;
                int* img_out_ptr = (int*)&cur_img(i, j);
                __m512i r_j = _mm512_loadu_epi32(tab_j);

                __m512 r_x = _mm512_sub_ps(_mm512_cvtepi32_ps(r_j), r_dim_2);
                __m512 r_ax = _mm512_abs_ps(r_x);
                
                __mmask16 mask_invert = _mm512_cmp_ps_mask(r_ay, r_ax, _CMP_GT_OQ);
                __m512 r_z = _mm512_mask_blend_ps(mask_invert, 
                                                  _mm512_div_ps(r_ay, r_ax), 
                                                  _mm512_div_ps(r_ax, r_ay));

                __m512 r_z2 = _mm512_mul_ps(r_z, r_z);
                __m512 r_den1 = _mm512_add_ps(r_z2, r_028); 
                __m512 r_path1 = _mm512_sub_ps(r_pi_2, _mm512_div_ps(r_z, r_den1));
                __m512 r_den2 = _mm512_fmadd_ps(r_028, r_z2, r_1);
                __m512 r_path2 = _mm512_div_ps(r_z, r_den2);

                __mmask16 mask_gt = _mm512_cmp_ps_mask(r_z, r_1, _CMP_GT_OQ);
                __m512 r_th = _mm512_mask_blend_ps(mask_gt, r_path2, r_path1);

                r_th = _mm512_mask_blend_ps(mask_invert, r_th, _mm512_sub_ps(r_pi_2, r_th));
                r_th = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(r_x, r_zero, _CMP_LT_OQ), r_th, _mm512_sub_ps(r_pi, r_th));
                r_th = _mm512_mask_blend_ps(mask_y_lt_0, r_th, _mm512_sub_ps(r_zero, r_th));

                __m512 r_angle = _mm512_add_ps(r_th, r_pi_base);
                __m512 r_div_fmod = _mm512_div_ps(r_angle, r_pi_4);
                __m512 r_trunc = _mm512_roundscale_ps(r_div_fmod, _MM_FROUND_TO_ZERO);
                __m512 r_fmod = _mm512_fnmadd_ps(r_trunc, r_pi_4, r_angle);

                __m512 r_ratio = _mm512_abs_ps(_mm512_div_ps(_mm512_sub_ps(r_fmod, r_pi_8), r_pi_8));

                __m512 r_r_float = _mm512_fmadd_ps(r_diff_r, r_ratio, r_blue_r);
                __m512 r_g_float = _mm512_fmadd_ps(r_diff_g, r_ratio, r_blue_g);
                __m512 r_b_float = _mm512_fmadd_ps(r_diff_b, r_ratio, r_blue_b);
                __m512 r_a_float = _mm512_fmadd_ps(r_diff_a, r_ratio, r_blue_a);

                __m512i r_result = rgba_simd(
                    _mm512_cvtps_epi32(r_r_float),
                    _mm512_cvtps_epi32(r_g_float),
                    _mm512_cvtps_epi32(r_b_float),
                    _mm512_cvtps_epi32(r_a_float)
                );

                _mm512_storeu_epi32(img_out_ptr, r_result);
            }
        }
        rotate(); // Slightly increase the base angle
    }
    return 0;
}

// ----------------------------------------------------------------------------
// ------------------------------------------------------------- SIMD VERSION 6
// ----------------------------------------------------------------------------

unsigned spin_compute_simd_v6u2(unsigned nb_iter) {
    int32_t tab_j[16];
    
    for (unsigned it = 1; it <= nb_iter; it++) {
        // --- POSITION A ---
        __m512 r_dim_2   = _mm512_set1_ps((float)DIM / 2.f);
        __m512 r_1       = _mm512_set1_ps(1.0f);
        __m512 r_028     = _mm512_set1_ps(0.28f);
        __m512 r_pi_2    = _mm512_set1_ps(M_PI_2sp);
        __m512 r_pi      = _mm512_set1_ps(M_PIsp);
        __m512 r_zero    = _mm512_setzero_ps();
        __m512 r_pi_base = _mm512_set1_ps(M_PIsp + base_angle);
        __m512 r_pi_4    = _mm512_set1_ps(M_PIsp / 4.f);
        __m512 r_pi_8    = _mm512_set1_ps(M_PIsp / 8.f);
        __m512 r_blue_r = _mm512_set1_ps((float)blue_r);
        __m512 r_blue_g = _mm512_set1_ps((float)blue_g);
        __m512 r_blue_b = _mm512_set1_ps((float)blue_b);
        __m512 r_blue_a = _mm512_set1_ps((float)blue_a);
        __m512 r_diff_r = _mm512_set1_ps((float)(yellow_r - blue_r));
        __m512 r_diff_g = _mm512_set1_ps((float)(yellow_g - blue_g));
        __m512 r_diff_b = _mm512_set1_ps((float)(yellow_b - blue_b));
        __m512 r_diff_a = _mm512_set1_ps((float)(yellow_a - blue_a));

        __m512i r_16 = _mm512_set1_epi32(16); // Vector filled with 16s

        for (unsigned i = 0; i < DIM; i++) {
            // --- POSITION B ---
            __m512i r_i = _mm512_set1_epi32(i);
            __m512 r_y = _mm512_sub_ps(r_dim_2, _mm512_cvtepi32_ps(r_i));
            __m512 r_ay = _mm512_abs_ps(r_y);
            __mmask16 mask_y_lt_0 = _mm512_cmp_ps_mask(r_y, r_zero, _CMP_LT_OQ);

            for (unsigned j = 0; j < DIM; j += 32) { // UNROLLED BY 2
                // --- POSITION C ---
                for (unsigned jj = 0; jj < 16; jj++) tab_j[jj] = j + jj;
                int* img_out_ptr = (int*)&cur_img(i, j);
                
                __m512i r_j_j0 = _mm512_loadu_epi32(tab_j);
                __m512i r_j_j1 = _mm512_add_epi32(r_j_j0, r_16); // Add 16 to create the second block!

                // --- BLOCK 0 (j0) ---
                __m512 r_x_j0 = _mm512_sub_ps(_mm512_cvtepi32_ps(r_j_j0), r_dim_2);
                __m512 r_ax_j0 = _mm512_abs_ps(r_x_j0);
                __mmask16 mask_invert_j0 = _mm512_cmp_ps_mask(r_ay, r_ax_j0, _CMP_GT_OQ);
                __m512 r_z_j0 = _mm512_mask_blend_ps(mask_invert_j0, _mm512_div_ps(r_ay, r_ax_j0), _mm512_div_ps(r_ax_j0, r_ay));
                __m512 r_z2_j0 = _mm512_mul_ps(r_z_j0, r_z_j0);
                __m512 r_den1_j0 = _mm512_add_ps(r_z2_j0, r_028); 
                __m512 r_path1_j0 = _mm512_sub_ps(r_pi_2, _mm512_div_ps(r_z_j0, r_den1_j0));
                __m512 r_den2_j0 = _mm512_fmadd_ps(r_028, r_z2_j0, r_1);
                __m512 r_path2_j0 = _mm512_div_ps(r_z_j0, r_den2_j0);
                __mmask16 mask_gt_j0 = _mm512_cmp_ps_mask(r_z_j0, r_1, _CMP_GT_OQ);
                __m512 r_th_j0 = _mm512_mask_blend_ps(mask_gt_j0, r_path2_j0, r_path1_j0);
                r_th_j0 = _mm512_mask_blend_ps(mask_invert_j0, r_th_j0, _mm512_sub_ps(r_pi_2, r_th_j0));
                r_th_j0 = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(r_x_j0, r_zero, _CMP_LT_OQ), r_th_j0, _mm512_sub_ps(r_pi, r_th_j0));
                r_th_j0 = _mm512_mask_blend_ps(mask_y_lt_0, r_th_j0, _mm512_sub_ps(r_zero, r_th_j0));
                __m512 r_angle_j0 = _mm512_add_ps(r_th_j0, r_pi_base);
                __m512 r_div_fmod_j0 = _mm512_div_ps(r_angle_j0, r_pi_4);
                __m512 r_trunc_j0 = _mm512_roundscale_ps(r_div_fmod_j0, _MM_FROUND_TO_ZERO);
                __m512 r_fmod_j0 = _mm512_fnmadd_ps(r_trunc_j0, r_pi_4, r_angle_j0);
                __m512 r_ratio_j0 = _mm512_abs_ps(_mm512_div_ps(_mm512_sub_ps(r_fmod_j0, r_pi_8), r_pi_8));
                __m512 r_r_float_j0 = _mm512_fmadd_ps(r_diff_r, r_ratio_j0, r_blue_r);
                __m512 r_g_float_j0 = _mm512_fmadd_ps(r_diff_g, r_ratio_j0, r_blue_g);
                __m512 r_b_float_j0 = _mm512_fmadd_ps(r_diff_b, r_ratio_j0, r_blue_b);
                __m512 r_a_float_j0 = _mm512_fmadd_ps(r_diff_a, r_ratio_j0, r_blue_a);
                __m512i r_result_j0 = rgba_simd(_mm512_cvtps_epi32(r_r_float_j0), _mm512_cvtps_epi32(r_g_float_j0), _mm512_cvtps_epi32(r_b_float_j0), _mm512_cvtps_epi32(r_a_float_j0));

                // --- BLOCK 1 (j1) ---
                __m512 r_x_j1 = _mm512_sub_ps(_mm512_cvtepi32_ps(r_j_j1), r_dim_2);
                __m512 r_ax_j1 = _mm512_abs_ps(r_x_j1);
                __mmask16 mask_invert_j1 = _mm512_cmp_ps_mask(r_ay, r_ax_j1, _CMP_GT_OQ);
                __m512 r_z_j1 = _mm512_mask_blend_ps(mask_invert_j1, _mm512_div_ps(r_ay, r_ax_j1), _mm512_div_ps(r_ax_j1, r_ay));
                __m512 r_z2_j1 = _mm512_mul_ps(r_z_j1, r_z_j1);
                __m512 r_den1_j1 = _mm512_add_ps(r_z2_j1, r_028); 
                __m512 r_path1_j1 = _mm512_sub_ps(r_pi_2, _mm512_div_ps(r_z_j1, r_den1_j1));
                __m512 r_den2_j1 = _mm512_fmadd_ps(r_028, r_z2_j1, r_1);
                __m512 r_path2_j1 = _mm512_div_ps(r_z_j1, r_den2_j1);
                __mmask16 mask_gt_j1 = _mm512_cmp_ps_mask(r_z_j1, r_1, _CMP_GT_OQ);
                __m512 r_th_j1 = _mm512_mask_blend_ps(mask_gt_j1, r_path2_j1, r_path1_j1);
                r_th_j1 = _mm512_mask_blend_ps(mask_invert_j1, r_th_j1, _mm512_sub_ps(r_pi_2, r_th_j1));
                r_th_j1 = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(r_x_j1, r_zero, _CMP_LT_OQ), r_th_j1, _mm512_sub_ps(r_pi, r_th_j1));
                r_th_j1 = _mm512_mask_blend_ps(mask_y_lt_0, r_th_j1, _mm512_sub_ps(r_zero, r_th_j1));
                __m512 r_angle_j1 = _mm512_add_ps(r_th_j1, r_pi_base);
                __m512 r_div_fmod_j1 = _mm512_div_ps(r_angle_j1, r_pi_4);
                __m512 r_trunc_j1 = _mm512_roundscale_ps(r_div_fmod_j1, _MM_FROUND_TO_ZERO);
                __m512 r_fmod_j1 = _mm512_fnmadd_ps(r_trunc_j1, r_pi_4, r_angle_j1);
                __m512 r_ratio_j1 = _mm512_abs_ps(_mm512_div_ps(_mm512_sub_ps(r_fmod_j1, r_pi_8), r_pi_8));
                __m512 r_r_float_j1 = _mm512_fmadd_ps(r_diff_r, r_ratio_j1, r_blue_r);
                __m512 r_g_float_j1 = _mm512_fmadd_ps(r_diff_g, r_ratio_j1, r_blue_g);
                __m512 r_b_float_j1 = _mm512_fmadd_ps(r_diff_b, r_ratio_j1, r_blue_b);
                __m512 r_a_float_j1 = _mm512_fmadd_ps(r_diff_a, r_ratio_j1, r_blue_a);
                __m512i r_result_j1 = rgba_simd(_mm512_cvtps_epi32(r_r_float_j1), _mm512_cvtps_epi32(r_g_float_j1), _mm512_cvtps_epi32(r_b_float_j1), _mm512_cvtps_epi32(r_a_float_j1));

                // Store both blocks sequentially to memory!
                _mm512_storeu_epi32(img_out_ptr, r_result_j0);
                _mm512_storeu_epi32(img_out_ptr + 16, r_result_j1);
            }
        }
        rotate(); 
    }
    return 0;
}

unsigned spin_compute_simd_v6u4(unsigned nb_iter) {
    int32_t tab_j[16];
    
    for (unsigned it = 1; it <= nb_iter; it++) {
        // --- POSITION A ---
        __m512 r_dim_2   = _mm512_set1_ps((float)DIM / 2.f);
        __m512 r_1       = _mm512_set1_ps(1.0f);
        __m512 r_028     = _mm512_set1_ps(0.28f);
        __m512 r_pi_2    = _mm512_set1_ps(M_PI_2sp);
        __m512 r_pi      = _mm512_set1_ps(M_PIsp);
        __m512 r_zero    = _mm512_setzero_ps();
        __m512 r_pi_base = _mm512_set1_ps(M_PIsp + base_angle);
        __m512 r_pi_4    = _mm512_set1_ps(M_PIsp / 4.f);
        __m512 r_pi_8    = _mm512_set1_ps(M_PIsp / 8.f);
        __m512 r_blue_r = _mm512_set1_ps((float)blue_r);
        __m512 r_blue_g = _mm512_set1_ps((float)blue_g);
        __m512 r_blue_b = _mm512_set1_ps((float)blue_b);
        __m512 r_blue_a = _mm512_set1_ps((float)blue_a);
        __m512 r_diff_r = _mm512_set1_ps((float)(yellow_r - blue_r));
        __m512 r_diff_g = _mm512_set1_ps((float)(yellow_g - blue_g));
        __m512 r_diff_b = _mm512_set1_ps((float)(yellow_b - blue_b));
        __m512 r_diff_a = _mm512_set1_ps((float)(yellow_a - blue_a));

        __m512i r_16 = _mm512_set1_epi32(16);
        __m512i r_32 = _mm512_set1_epi32(32);
        __m512i r_48 = _mm512_set1_epi32(48);

        for (unsigned i = 0; i < DIM; i++) {
            // --- POSITION B ---
            __m512i r_i = _mm512_set1_epi32(i);
            __m512 r_y = _mm512_sub_ps(r_dim_2, _mm512_cvtepi32_ps(r_i));
            __m512 r_ay = _mm512_abs_ps(r_y);
            __mmask16 mask_y_lt_0 = _mm512_cmp_ps_mask(r_y, r_zero, _CMP_LT_OQ);

            for (unsigned j = 0; j < DIM; j += 64) { // UNROLLED BY 4
                // --- POSITION C ---
                for (unsigned jj = 0; jj < 16; jj++) tab_j[jj] = j + jj;
                int* img_out_ptr = (int*)&cur_img(i, j);
                
                __m512i r_j_j0 = _mm512_loadu_epi32(tab_j);
                __m512i r_j_j1 = _mm512_add_epi32(r_j_j0, r_16);
                __m512i r_j_j2 = _mm512_add_epi32(r_j_j0, r_32);
                __m512i r_j_j3 = _mm512_add_epi32(r_j_j0, r_48);

                // --- BLOCK 0 (j0) ---
                __m512 r_x_j0 = _mm512_sub_ps(_mm512_cvtepi32_ps(r_j_j0), r_dim_2);
                __m512 r_ax_j0 = _mm512_abs_ps(r_x_j0);
                __mmask16 mask_invert_j0 = _mm512_cmp_ps_mask(r_ay, r_ax_j0, _CMP_GT_OQ);
                __m512 r_z_j0 = _mm512_mask_blend_ps(mask_invert_j0, _mm512_div_ps(r_ay, r_ax_j0), _mm512_div_ps(r_ax_j0, r_ay));
                __m512 r_z2_j0 = _mm512_mul_ps(r_z_j0, r_z_j0);
                __m512 r_den1_j0 = _mm512_add_ps(r_z2_j0, r_028); 
                __m512 r_path1_j0 = _mm512_sub_ps(r_pi_2, _mm512_div_ps(r_z_j0, r_den1_j0));
                __m512 r_den2_j0 = _mm512_fmadd_ps(r_028, r_z2_j0, r_1);
                __m512 r_path2_j0 = _mm512_div_ps(r_z_j0, r_den2_j0);
                __mmask16 mask_gt_j0 = _mm512_cmp_ps_mask(r_z_j0, r_1, _CMP_GT_OQ);
                __m512 r_th_j0 = _mm512_mask_blend_ps(mask_gt_j0, r_path2_j0, r_path1_j0);
                r_th_j0 = _mm512_mask_blend_ps(mask_invert_j0, r_th_j0, _mm512_sub_ps(r_pi_2, r_th_j0));
                r_th_j0 = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(r_x_j0, r_zero, _CMP_LT_OQ), r_th_j0, _mm512_sub_ps(r_pi, r_th_j0));
                r_th_j0 = _mm512_mask_blend_ps(mask_y_lt_0, r_th_j0, _mm512_sub_ps(r_zero, r_th_j0));
                __m512 r_angle_j0 = _mm512_add_ps(r_th_j0, r_pi_base);
                __m512 r_div_fmod_j0 = _mm512_div_ps(r_angle_j0, r_pi_4);
                __m512 r_trunc_j0 = _mm512_roundscale_ps(r_div_fmod_j0, _MM_FROUND_TO_ZERO);
                __m512 r_fmod_j0 = _mm512_fnmadd_ps(r_trunc_j0, r_pi_4, r_angle_j0);
                __m512 r_ratio_j0 = _mm512_abs_ps(_mm512_div_ps(_mm512_sub_ps(r_fmod_j0, r_pi_8), r_pi_8));
                __m512 r_r_float_j0 = _mm512_fmadd_ps(r_diff_r, r_ratio_j0, r_blue_r);
                __m512 r_g_float_j0 = _mm512_fmadd_ps(r_diff_g, r_ratio_j0, r_blue_g);
                __m512 r_b_float_j0 = _mm512_fmadd_ps(r_diff_b, r_ratio_j0, r_blue_b);
                __m512 r_a_float_j0 = _mm512_fmadd_ps(r_diff_a, r_ratio_j0, r_blue_a);
                __m512i r_result_j0 = rgba_simd(_mm512_cvtps_epi32(r_r_float_j0), _mm512_cvtps_epi32(r_g_float_j0), _mm512_cvtps_epi32(r_b_float_j0), _mm512_cvtps_epi32(r_a_float_j0));

                // --- BLOCK 1 (j1) ---
                __m512 r_x_j1 = _mm512_sub_ps(_mm512_cvtepi32_ps(r_j_j1), r_dim_2);
                __m512 r_ax_j1 = _mm512_abs_ps(r_x_j1);
                __mmask16 mask_invert_j1 = _mm512_cmp_ps_mask(r_ay, r_ax_j1, _CMP_GT_OQ);
                __m512 r_z_j1 = _mm512_mask_blend_ps(mask_invert_j1, _mm512_div_ps(r_ay, r_ax_j1), _mm512_div_ps(r_ax_j1, r_ay));
                __m512 r_z2_j1 = _mm512_mul_ps(r_z_j1, r_z_j1);
                __m512 r_den1_j1 = _mm512_add_ps(r_z2_j1, r_028); 
                __m512 r_path1_j1 = _mm512_sub_ps(r_pi_2, _mm512_div_ps(r_z_j1, r_den1_j1));
                __m512 r_den2_j1 = _mm512_fmadd_ps(r_028, r_z2_j1, r_1);
                __m512 r_path2_j1 = _mm512_div_ps(r_z_j1, r_den2_j1);
                __mmask16 mask_gt_j1 = _mm512_cmp_ps_mask(r_z_j1, r_1, _CMP_GT_OQ);
                __m512 r_th_j1 = _mm512_mask_blend_ps(mask_gt_j1, r_path2_j1, r_path1_j1);
                r_th_j1 = _mm512_mask_blend_ps(mask_invert_j1, r_th_j1, _mm512_sub_ps(r_pi_2, r_th_j1));
                r_th_j1 = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(r_x_j1, r_zero, _CMP_LT_OQ), r_th_j1, _mm512_sub_ps(r_pi, r_th_j1));
                r_th_j1 = _mm512_mask_blend_ps(mask_y_lt_0, r_th_j1, _mm512_sub_ps(r_zero, r_th_j1));
                __m512 r_angle_j1 = _mm512_add_ps(r_th_j1, r_pi_base);
                __m512 r_div_fmod_j1 = _mm512_div_ps(r_angle_j1, r_pi_4);
                __m512 r_trunc_j1 = _mm512_roundscale_ps(r_div_fmod_j1, _MM_FROUND_TO_ZERO);
                __m512 r_fmod_j1 = _mm512_fnmadd_ps(r_trunc_j1, r_pi_4, r_angle_j1);
                __m512 r_ratio_j1 = _mm512_abs_ps(_mm512_div_ps(_mm512_sub_ps(r_fmod_j1, r_pi_8), r_pi_8));
                __m512 r_r_float_j1 = _mm512_fmadd_ps(r_diff_r, r_ratio_j1, r_blue_r);
                __m512 r_g_float_j1 = _mm512_fmadd_ps(r_diff_g, r_ratio_j1, r_blue_g);
                __m512 r_b_float_j1 = _mm512_fmadd_ps(r_diff_b, r_ratio_j1, r_blue_b);
                __m512 r_a_float_j1 = _mm512_fmadd_ps(r_diff_a, r_ratio_j1, r_blue_a);
                __m512i r_result_j1 = rgba_simd(_mm512_cvtps_epi32(r_r_float_j1), _mm512_cvtps_epi32(r_g_float_j1), _mm512_cvtps_epi32(r_b_float_j1), _mm512_cvtps_epi32(r_a_float_j1));

                // --- BLOCK 2 (j2) ---
                __m512 r_x_j2 = _mm512_sub_ps(_mm512_cvtepi32_ps(r_j_j2), r_dim_2);
                __m512 r_ax_j2 = _mm512_abs_ps(r_x_j2);
                __mmask16 mask_invert_j2 = _mm512_cmp_ps_mask(r_ay, r_ax_j2, _CMP_GT_OQ);
                __m512 r_z_j2 = _mm512_mask_blend_ps(mask_invert_j2, _mm512_div_ps(r_ay, r_ax_j2), _mm512_div_ps(r_ax_j2, r_ay));
                __m512 r_z2_j2 = _mm512_mul_ps(r_z_j2, r_z_j2);
                __m512 r_den1_j2 = _mm512_add_ps(r_z2_j2, r_028); 
                __m512 r_path1_j2 = _mm512_sub_ps(r_pi_2, _mm512_div_ps(r_z_j2, r_den1_j2));
                __m512 r_den2_j2 = _mm512_fmadd_ps(r_028, r_z2_j2, r_1);
                __m512 r_path2_j2 = _mm512_div_ps(r_z_j2, r_den2_j2);
                __mmask16 mask_gt_j2 = _mm512_cmp_ps_mask(r_z_j2, r_1, _CMP_GT_OQ);
                __m512 r_th_j2 = _mm512_mask_blend_ps(mask_gt_j2, r_path2_j2, r_path1_j2);
                r_th_j2 = _mm512_mask_blend_ps(mask_invert_j2, r_th_j2, _mm512_sub_ps(r_pi_2, r_th_j2));
                r_th_j2 = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(r_x_j2, r_zero, _CMP_LT_OQ), r_th_j2, _mm512_sub_ps(r_pi, r_th_j2));
                r_th_j2 = _mm512_mask_blend_ps(mask_y_lt_0, r_th_j2, _mm512_sub_ps(r_zero, r_th_j2));
                __m512 r_angle_j2 = _mm512_add_ps(r_th_j2, r_pi_base);
                __m512 r_div_fmod_j2 = _mm512_div_ps(r_angle_j2, r_pi_4);
                __m512 r_trunc_j2 = _mm512_roundscale_ps(r_div_fmod_j2, _MM_FROUND_TO_ZERO);
                __m512 r_fmod_j2 = _mm512_fnmadd_ps(r_trunc_j2, r_pi_4, r_angle_j2);
                __m512 r_ratio_j2 = _mm512_abs_ps(_mm512_div_ps(_mm512_sub_ps(r_fmod_j2, r_pi_8), r_pi_8));
                __m512 r_r_float_j2 = _mm512_fmadd_ps(r_diff_r, r_ratio_j2, r_blue_r);
                __m512 r_g_float_j2 = _mm512_fmadd_ps(r_diff_g, r_ratio_j2, r_blue_g);
                __m512 r_b_float_j2 = _mm512_fmadd_ps(r_diff_b, r_ratio_j2, r_blue_b);
                __m512 r_a_float_j2 = _mm512_fmadd_ps(r_diff_a, r_ratio_j2, r_blue_a);
                __m512i r_result_j2 = rgba_simd(_mm512_cvtps_epi32(r_r_float_j2), _mm512_cvtps_epi32(r_g_float_j2), _mm512_cvtps_epi32(r_b_float_j2), _mm512_cvtps_epi32(r_a_float_j2));

                // --- BLOCK 3 (j3) ---
                __m512 r_x_j3 = _mm512_sub_ps(_mm512_cvtepi32_ps(r_j_j3), r_dim_2);
                __m512 r_ax_j3 = _mm512_abs_ps(r_x_j3);
                __mmask16 mask_invert_j3 = _mm512_cmp_ps_mask(r_ay, r_ax_j3, _CMP_GT_OQ);
                __m512 r_z_j3 = _mm512_mask_blend_ps(mask_invert_j3, _mm512_div_ps(r_ay, r_ax_j3), _mm512_div_ps(r_ax_j3, r_ay));
                __m512 r_z2_j3 = _mm512_mul_ps(r_z_j3, r_z_j3);
                __m512 r_den1_j3 = _mm512_add_ps(r_z2_j3, r_028); 
                __m512 r_path1_j3 = _mm512_sub_ps(r_pi_2, _mm512_div_ps(r_z_j3, r_den1_j3));
                __m512 r_den2_j3 = _mm512_fmadd_ps(r_028, r_z2_j3, r_1);
                __m512 r_path2_j3 = _mm512_div_ps(r_z_j3, r_den2_j3);
                __mmask16 mask_gt_j3 = _mm512_cmp_ps_mask(r_z_j3, r_1, _CMP_GT_OQ);
                __m512 r_th_j3 = _mm512_mask_blend_ps(mask_gt_j3, r_path2_j3, r_path1_j3);
                r_th_j3 = _mm512_mask_blend_ps(mask_invert_j3, r_th_j3, _mm512_sub_ps(r_pi_2, r_th_j3));
                r_th_j3 = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(r_x_j3, r_zero, _CMP_LT_OQ), r_th_j3, _mm512_sub_ps(r_pi, r_th_j3));
                r_th_j3 = _mm512_mask_blend_ps(mask_y_lt_0, r_th_j3, _mm512_sub_ps(r_zero, r_th_j3));
                __m512 r_angle_j3 = _mm512_add_ps(r_th_j3, r_pi_base);
                __m512 r_div_fmod_j3 = _mm512_div_ps(r_angle_j3, r_pi_4);
                __m512 r_trunc_j3 = _mm512_roundscale_ps(r_div_fmod_j3, _MM_FROUND_TO_ZERO);
                __m512 r_fmod_j3 = _mm512_fnmadd_ps(r_trunc_j3, r_pi_4, r_angle_j3);
                __m512 r_ratio_j3 = _mm512_abs_ps(_mm512_div_ps(_mm512_sub_ps(r_fmod_j3, r_pi_8), r_pi_8));
                __m512 r_r_float_j3 = _mm512_fmadd_ps(r_diff_r, r_ratio_j3, r_blue_r);
                __m512 r_g_float_j3 = _mm512_fmadd_ps(r_diff_g, r_ratio_j3, r_blue_g);
                __m512 r_b_float_j3 = _mm512_fmadd_ps(r_diff_b, r_ratio_j3, r_blue_b);
                __m512 r_a_float_j3 = _mm512_fmadd_ps(r_diff_a, r_ratio_j3, r_blue_a);
                __m512i r_result_j3 = rgba_simd(_mm512_cvtps_epi32(r_r_float_j3), _mm512_cvtps_epi32(r_g_float_j3), _mm512_cvtps_epi32(r_b_float_j3), _mm512_cvtps_epi32(r_a_float_j3));

                // Store all 4 blocks sequentially to memory!
                _mm512_storeu_epi32(img_out_ptr, r_result_j0);
                _mm512_storeu_epi32(img_out_ptr + 16, r_result_j1);
                _mm512_storeu_epi32(img_out_ptr + 32, r_result_j2);
                _mm512_storeu_epi32(img_out_ptr + 48, r_result_j3);
            }
        }
        rotate(); 
    }
    return 0;
}



#endif /* __AVX512F__ */

#endif /* ENABLE_VECTO */


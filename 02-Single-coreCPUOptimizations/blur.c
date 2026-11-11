/*
Wilson Cedric Genevieve
LAB2 PACC Course
Nov 26
*/

#include "easypap.h"

#include <omp.h>

///////////////////////////// Sequential version (tiled)
// Suggested cmdline(s):
// ./run -l data/img/1024.png -k blur -v seq -si
//
int blur_do_tile_default (int x, int y, int width, int height)
{
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++) {
      unsigned r = 0, g = 0, b = 0, a = 0, n = 0;

      int i_d = (i > 0) ? i - 1 : i;
      int i_f = (i < DIM - 1) ? i + 1 : i;
      int j_d = (j > 0) ? j - 1 : j;
      int j_f = (j < DIM - 1) ? j + 1 : j;

      for (int yloc = i_d; yloc <= i_f; yloc++)
        for (int xloc = j_d; xloc <= j_f; xloc++) {
          unsigned c = cur_img (yloc, xloc);
          r += ezv_c2r (c);
          g += ezv_c2g (c);
          b += ezv_c2b (c);
          a += ezv_c2a (c);
          n += 1;
        }

      r /= n;
      g /= n;
      b /= n;
      a /= n;

      next_img (i, j) = ezv_rgba (r, g, b, a);
    }

  return 0;
}

///////////////////////////// 2.1 Without Border
// Optimisation: Removed border checks to assume a guaranteed 3x3 neighborhood for every pixel.
int blur_do_tile_default_nb (int x, int y, int width, int height)
{
  for (int i = y+1; i < y + height-1; i++)
    for (int j = x+1; j < x + width-1; j++) { // forgetting these minor adjustments causes seg fault
      unsigned r = 0, g = 0, b = 0, a = 0, n = 0;

      // Hardcoded loop bounds to directly sweep from i-1 to i+1 and j-1 to j+1 (and don't bother checking for borders)
      for (int yloc = i-1; yloc <= i+1; yloc++)
        for (int xloc = j-1; xloc <= j+1; xloc++) {
          unsigned c = cur_img (yloc, xloc);
          r += ezv_c2r (c);
          g += ezv_c2g (c);
          b += ezv_c2b (c);
          a += ezv_c2a (c);
          n += 1;
        }

      r /= n;
      g /= n;
      b /= n;
      a /= n;

      next_img (i, j) = ezv_rgba (r, g, b, a);
    }

  return 0;
}


///////////////////////////// 2.2 Loop Unrolling on X-axis
int blur_do_tile_optim1 (int x, int y, int width, int height)
{
  for (int i = y+1; i < y + height-1; i++)
    for (int j = x+1; j < x + width-1; j++) { 
      unsigned r = 0, g = 0, b = 0, a = 0; // we dont need n anymore since it's hardcoded

      // 
      for (int yloc = i-1; yloc <= i+1; yloc++){
        // left pixel
        unsigned c1 = cur_img (yloc, j-1);
	r += ezv_c2r(c1); g += ezv_c2g(c1); b += ezv_c2b(c1); a += ezv_c2a(c1);

	// Center pixel
	unsigned c2 = cur_img(yloc, j);
	r += ezv_c2r(c2); g += ezv_c2g(c2); b += ezv_c2b(c2); a += ezv_c2a(c2);

	// Right pixel
	unsigned c3 = cur_img(yloc, j + 1);
	r += ezv_c2r(c3); g += ezv_c2g(c3); b += ezv_c2b(c3); a += ezv_c2a(c3);
      }

      r /= 9;
      g /= 9;
      b /= 9;
      a /= 9;

      next_img (i, j) = ezv_rgba (r, g, b, a);
    }

  return 0;
}

///////////////////////////// 2.3 Loop Unrolling on Y-axis
int blur_do_tile_optim2 (int x, int y, int width, int height)
{
  for (int i = y + 1; i < y + height - 1; i++) {
    for (int j = x + 1; j < x + width - 1; j++) {
        unsigned r = 0, g = 0, b = 0, a = 0;

        // Top row (i-1)
        unsigned c1 = cur_img(i - 1, j - 1);
        r += ezv_c2r(c1); g += ezv_c2g(c1); b += ezv_c2b(c1); a += ezv_c2a(c1);
        unsigned c2 = cur_img(i - 1, j);
        r += ezv_c2r(c2); g += ezv_c2g(c2); b += ezv_c2b(c2); a += ezv_c2a(c2);
        unsigned c3 = cur_img(i - 1, j + 1);
        r += ezv_c2r(c3); g += ezv_c2g(c3); b += ezv_c2b(c3); a += ezv_c2a(c3);

        // Middle row (i)
        unsigned c4 = cur_img(i, j - 1);
        r += ezv_c2r(c4); g += ezv_c2g(c4); b += ezv_c2b(c4); a += ezv_c2a(c4);
        unsigned c5 = cur_img(i, j);
        r += ezv_c2r(c5); g += ezv_c2g(c5); b += ezv_c2b(c5); a += ezv_c2a(c5);
        unsigned c6 = cur_img(i, j + 1);
        r += ezv_c2r(c6); g += ezv_c2g(c6); b += ezv_c2b(c6); a += ezv_c2a(c6);

        // Bottom row (i+1)
        unsigned c7 = cur_img(i + 1, j - 1);
        r += ezv_c2r(c7); g += ezv_c2g(c7); b += ezv_c2b(c7); a += ezv_c2a(c7);
        unsigned c8 = cur_img(i + 1, j);
        r += ezv_c2r(c8); g += ezv_c2g(c8); b += ezv_c2b(c8); a += ezv_c2a(c8);
        unsigned c9 = cur_img(i + 1, j + 1);
        r += ezv_c2r(c9); g += ezv_c2g(c9); b += ezv_c2b(c9); a += ezv_c2a(c9);

        r /= 9; g /= 9; b /= 9; a /= 9;
        next_img(i, j) = ezv_rgba(r, g, b, a);
    }
}

  return 0;
}


///////////////////////////// 2.4 Inline Functiomn Calls
// The concept opf the optimisation here is that too many fucntion calls slow us down, so if we can avoid fucntion calls, let's avoid them!
int blur_do_tile_optim3 (int x, int y, int width, int height)
{

  for (int i = y + 1; i < y + height - 1; i++) {
    for (int j = x + 1; j < x + width - 1; j++) {
        unsigned r = 0, g = 0, b = 0, a = 0;

	// The idea here is to do manual inline of ezv_c2r/g/b/a from ezv_rgba.h 
	// This avoids 36 expensive function call jumps per pixel under -O0

        // Top row (i-1)
        unsigned c1 = cur_img(i - 1, j - 1);
        r += (uint8_t)c1; g += (uint8_t)(c1 >> 8); b += (uint8_t)(c1 >> 16); a += (uint8_t)(c1 >> 24);
        unsigned c2 = cur_img(i - 1, j);
        r += (uint8_t)c2; g += (uint8_t)(c2 >> 8); b += (uint8_t)(c2 >> 16); a += (uint8_t)(c2 >> 24);
        unsigned c3 = cur_img(i - 1, j + 1);
        r += (uint8_t)c3; g += (uint8_t)(c3 >> 8); b += (uint8_t)(c3 >> 16); a += (uint8_t)(c3 >> 24);

        // Middle row (i)
        unsigned c4 = cur_img(i, j - 1);
        r += (uint8_t)c4; g += (uint8_t)(c4 >> 8); b += (uint8_t)(c4 >> 16); a += (uint8_t)(c4 >> 24);
        unsigned c5 = cur_img(i, j);
        r += (uint8_t)c5; g += (uint8_t)(c5 >> 8); b += (uint8_t)(c5 >> 16); a += (uint8_t)(c5 >> 24);
        unsigned c6 = cur_img(i, j + 1);
        r += (uint8_t)c6; g += (uint8_t)(c6 >> 8); b += (uint8_t)(c6 >> 16); a += (uint8_t)(c6 >> 24);

        // Bottom row (i+1)
        unsigned c7 = cur_img(i + 1, j - 1);
        r += (uint8_t)c7; g += (uint8_t)(c7 >> 8); b += (uint8_t)(c7 >> 16); a += (uint8_t)(c7 >> 24);
        unsigned c8 = cur_img(i + 1, j);
        r += (uint8_t)c8; g += (uint8_t)(c8 >> 8); b += (uint8_t)(c8 >> 16); a += (uint8_t)(c8 >> 24);
        unsigned c9 = cur_img(i + 1, j + 1);
        r += (uint8_t)c9; g += (uint8_t)(c9 >> 8); b += (uint8_t)(c9 >> 16); a += (uint8_t)(c9 >> 24);

        r /= 9; g /= 9; b /= 9; a /= 9;
        
	// Manual inline of ezv_rgba to repack the pixel using bitwise shifts and ORs to reconstruct the pixel
        next_img(i, j) = r | (g << 8) | (b << 16) | (a << 24);
    }
}

  return 0;
}


///////////////////////////// 2.5 Variables Rotation (on the X-axis)
// This optimization reuses data that has already been loaded from memory.
// By passing the column totals down the line (Left <- Center, Center <- Right), we only need to fetch the brand new Right column from RAM for each pixel.
// This eliminates redundant reads and significantly speeds up the loop!
int blur_do_tile_optim4 (int x, int y, int width, int height)
{

  for (int i = y + 1; i < y + height - 1; i++) {
    
    // 1. Calculate Left (x) and Center (x+1) columns BEFORE the j loop
    unsigned r_left = 0, g_left = 0, b_left = 0, a_left = 0;
    unsigned r_center = 0, g_center = 0, b_center = 0, a_center = 0;

    // Fetch the 3 pixels for the Left column (column x)
    unsigned c1 = cur_img(i - 1, x);
    r_left += (uint8_t)c1; g_left += (uint8_t)(c1 >> 8); b_left += (uint8_t)(c1 >> 16); a_left += (uint8_t)(c1 >> 24);
    unsigned c4 = cur_img(i, x);
    r_left += (uint8_t)c4; g_left += (uint8_t)(c4 >> 8); b_left += (uint8_t)(c4 >> 16); a_left += (uint8_t)(c4 >> 24);
    unsigned c7 = cur_img(i + 1, x);
    r_left += (uint8_t)c7; g_left += (uint8_t)(c7 >> 8); b_left += (uint8_t)(c7 >> 16); a_left += (uint8_t)(c7 >> 24);

    // Fetch the 3 pixels for the Center column (column x + 1)
    unsigned c2 = cur_img(i - 1, x + 1);
    r_center += (uint8_t)c2; g_center += (uint8_t)(c2 >> 8); b_center += (uint8_t)(c2 >> 16); a_center += (uint8_t)(c2 >> 24);
    unsigned c5 = cur_img(i, x + 1);
    r_center += (uint8_t)c5; g_center += (uint8_t)(c5 >> 8); b_center += (uint8_t)(c5 >> 16); a_center += (uint8_t)(c5 >> 24);
    unsigned c8 = cur_img(i + 1, x + 1);
    r_center += (uint8_t)c8; g_center += (uint8_t)(c8 >> 8); b_center += (uint8_t)(c8 >> 16); a_center += (uint8_t)(c8 >> 24);

    // 2. THE MAIN LOOP: Now we sweep across the row
    for (int j = x + 1; j < x + width - 1; j++) {
        unsigned r_right = 0, g_right = 0, b_right = 0, a_right = 0;

        // Fetch ONLY the brand new Right column (column j + 1)
        unsigned c3 = cur_img(i - 1, j + 1);
        r_right += (uint8_t)c3; g_right += (uint8_t)(c3 >> 8); b_right += (uint8_t)(c3 >> 16); a_right += (uint8_t)(c3 >> 24);
        unsigned c6 = cur_img(i, j + 1);
        r_right += (uint8_t)c6; g_right += (uint8_t)(c6 >> 8); b_right += (uint8_t)(c6 >> 16); a_right += (uint8_t)(c6 >> 24);
        unsigned c9 = cur_img(i + 1, j + 1);
        r_right += (uint8_t)c9; g_right += (uint8_t)(c9 >> 8); b_right += (uint8_t)(c9 >> 16); a_right += (uint8_t)(c9 >> 24);

        // Add the three columns together
        unsigned r = r_left + r_center + r_right;
        unsigned g = g_left + g_center + g_right;
        unsigned b = b_left + b_center + b_right;
        unsigned a = a_left + a_center + a_right;

        r /= 9; g /= 9; b /= 9; a /= 9;
        
        next_img(i, j) = r | (g << 8) | (b << 16) | (a << 24);

        // 3. ROTATE: Shift the conveyor belt for the next step of the loop!
        r_left = r_center; g_left = g_center; b_left = b_center; a_left = a_center;
        r_center = r_right; g_center = g_right; b_center = b_right; a_center = a_right;
    }
} 

  return 0;
}


///////////////////////////// 2.6 Manage Borders
// The concept here is that our fast loops in optim4 completely ignore the 1-pixel border around the very outside of the image (rows 0 and height-1, columns 0 and width-1). Because of this, our output image won't perfectly match the expected hash
// so we just need to cater for this at the end. I'll leave eveything untouched and just add dthe border logic at the end of the function
int blur_do_tile_optim5 (int x, int y, int width, int height)
{

  for (int i = y + 1; i < y + height - 1; i++) {
    
    // 1. Calculate Left (x) and Center (x+1) columns BEFORE the j loop
    unsigned r_left = 0, g_left = 0, b_left = 0, a_left = 0;
    unsigned r_center = 0, g_center = 0, b_center = 0, a_center = 0;

    // Fetch the 3 pixels for the Left column (column x)
    unsigned c1 = cur_img(i - 1, x);
    r_left += (uint8_t)c1; g_left += (uint8_t)(c1 >> 8); b_left += (uint8_t)(c1 >> 16); a_left += (uint8_t)(c1 >> 24);
    unsigned c4 = cur_img(i, x);
    r_left += (uint8_t)c4; g_left += (uint8_t)(c4 >> 8); b_left += (uint8_t)(c4 >> 16); a_left += (uint8_t)(c4 >> 24);
    unsigned c7 = cur_img(i + 1, x);
    r_left += (uint8_t)c7; g_left += (uint8_t)(c7 >> 8); b_left += (uint8_t)(c7 >> 16); a_left += (uint8_t)(c7 >> 24);

    // Fetch the 3 pixels for the Center column (column x + 1)
    unsigned c2 = cur_img(i - 1, x + 1);
    r_center += (uint8_t)c2; g_center += (uint8_t)(c2 >> 8); b_center += (uint8_t)(c2 >> 16); a_center += (uint8_t)(c2 >> 24);
    unsigned c5 = cur_img(i, x + 1);
    r_center += (uint8_t)c5; g_center += (uint8_t)(c5 >> 8); b_center += (uint8_t)(c5 >> 16); a_center += (uint8_t)(c5 >> 24);
    unsigned c8 = cur_img(i + 1, x + 1);
    r_center += (uint8_t)c8; g_center += (uint8_t)(c8 >> 8); b_center += (uint8_t)(c8 >> 16); a_center += (uint8_t)(c8 >> 24);

    // 2. THE MAIN LOOP: Now we sweep across the row
    for (int j = x + 1; j < x + width - 1; j++) {
        unsigned r_right = 0, g_right = 0, b_right = 0, a_right = 0;

        // Fetch ONLY the brand new Right column (column j + 1)
        unsigned c3 = cur_img(i - 1, j + 1);
        r_right += (uint8_t)c3; g_right += (uint8_t)(c3 >> 8); b_right += (uint8_t)(c3 >> 16); a_right += (uint8_t)(c3 >> 24);
        unsigned c6 = cur_img(i, j + 1);
        r_right += (uint8_t)c6; g_right += (uint8_t)(c6 >> 8); b_right += (uint8_t)(c6 >> 16); a_right += (uint8_t)(c6 >> 24);
        unsigned c9 = cur_img(i + 1, j + 1);
        r_right += (uint8_t)c9; g_right += (uint8_t)(c9 >> 8); b_right += (uint8_t)(c9 >> 16); a_right += (uint8_t)(c9 >> 24);

        // Add the three columns together
        unsigned r = r_left + r_center + r_right;
        unsigned g = g_left + g_center + g_right;
        unsigned b = b_left + b_center + b_right;
        unsigned a = a_left + a_center + a_right;

        r /= 9; g /= 9; b /= 9; a /= 9;
        
        next_img(i, j) = r | (g << 8) | (b << 16) | (a << 24);

        // 3. ROTATE: Shift the conveyor belt for the next step of the loop!
        r_left = r_center; g_left = g_center; b_left = b_center; a_left = a_center;
        r_center = r_right; g_center = g_right; b_center = b_right; a_center = a_right;
    }
}


// FINAL STEP: Separate pass just for the edges
    for (int i = y; i < y + height; i++) {
        for (int j = x; j < x + width; j++) {
            // If we are safely inside the center, skip! Our fast loop already did this.
            if (i > y && i < y + height - 1 && j > x && j < x + width - 1) {
                continue;
            }

            // Otherwise, we are on a border. Use the original safe code:
            unsigned r = 0, g = 0, b = 0, a = 0, n = 0;
            int i_d = (i > 0) ? i - 1 : i;
            int i_f = (i < DIM - 1) ? i + 1 : i;
            int j_d = (j > 0) ? j - 1 : j;
            int j_f = (j < DIM - 1) ? j + 1 : j;

            for (int yloc = i_d; yloc <= i_f; yloc++) {
                for (int xloc = j_d; xloc <= j_f; xloc++) {
                    unsigned c = cur_img(yloc, xloc);
                    r += ezv_c2r(c); g += ezv_c2g(c); b += ezv_c2b(c); a += ezv_c2a(c);
                    n += 1;
                }
            }
            next_img(i, j) = ezv_rgba(r / n, g / n, b / n, a / n);
        }
    }
 

  return 0;
}



///////////////////////////// Sequential version (tiled)
// Suggested cmdline(s):
// ./run -l data/img/1024.png -k blur -v seq
//
unsigned blur_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    do_tile (0, 0, DIM, DIM);

    swap_images ();
  }

  return 0;
}

///////////////////////////// Tiled sequential version (tiled)
// Suggested cmdline(s):
// ./run -l data/img/1024.png -k blur -v tiled -ts 32 -m si
//
unsigned blur_compute_tiled (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        do_tile (x, y, TILE_W, TILE_H);

    swap_images ();
  }

  return 0;
}

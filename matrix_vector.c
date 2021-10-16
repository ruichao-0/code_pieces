/*  Matrix vector multiplication 
 *  Author: Ruichao Jiang on 2021-10-13
 *--------------------------------------
 *
 * Two vector-multiplied-by-matrix functions. First written using pure C; second using C + ARM NEON intrinsics
 * Hardware: MacBook Pro 2012 running QEMU ARM emulator.
 *
 * Assumption: Matrix is assumed to be m by n, where m and n are multiples of 4. Leftover case was not dealt with.
 *             Pure C assumes a row-major matrix; NEON assumes a column-major matrix.
 *             Vector-matrix compatibility is user's responsibility.
 *
 * Result: For the test provided in the main(), pure C took 0.000526s, NEON intrinsics took 0.000327s.
 *         However, there are some variations on the result. Sometimes, very rarely, especially right after compilling, the result reverses. 
 *         This program is barely runnable, poorly optimized, considered that I learnt these embedded thing after getting your interview question. 
 *         But it already shows some improvement with NEON.
 */

#include <stdio.h>
#include <time.h>
#include <arm_neon.h>

#define ROUNDUP 64

void matrix_vector_multiplication (const int8_t *matrix, int32_t num_rows, int32_t num_columns, const int16_t *input, int16_t *output);

void matrix_vector_multiplication_arm (const int8_t *matrix, int32_t num_rows, int32_t num_columns, const int16_t *input, int16_t *output);

int main(int argc, const char * argv[]) {
  //const int8_t matrix_column_major[16] = {56, 2, 3, 1, 34, 3, 5, 2, 63, 4, 6, 5, 23, 5, 7, 3};
  //const int8_t matrix_row_major[16] = {56, 34, 63, 23, 2, 3, 4, 5, 3, 5, 6, 7, 1, 2, 5, 3};
  //const int16_t input[4] = {1000, 3000, 4000, 5000};
  int8_t matrix_column_major[10000];
  int8_t matrix_row_major[10000];
  int16_t input[100];
  
  for(int i = 0; i < 10000; ++i){
    matrix_column_major[i] = 1;
    matrix_row_major[i] = 1;
  }
  
  for(int i = 0; i < 100; ++i) {
    input[i] = i;
  }
  
  int16_t output[100];
  
  clock_t begin = clock();
  matrix_vector_multiplication(matrix_row_major, 100, 100, input, output);
  clock_t end = clock();
 
  printf("(%d,%d,%d,%d)\n", output[0], output[1], output[2], output[3]);
  printf("Plain C takes %f seconds\n", (double) (end - begin) / CLOCKS_PER_SEC);
  
  begin = clock();
  matrix_vector_multiplication_arm(matrix_column_major, 100, 100, input, output);
  end = clock();
  
  printf("(%d,%d,%d,%d)\n", output[0], output[1], output[2], output[3]);
  printf("ARM intrinsic takes %f seconds\n", (double) (end - begin) / CLOCKS_PER_SEC);
  
  return 0;
}

void matrix_vector_multiplication (const int8_t *matirx, int32_t num_rows, int32_t num_columns, const int16_t *input, int16_t *output){
  int32_t temp;
  for (int32_t i = 0; i < num_rows; ++i){
    temp = 0;
    for (int32_t j = 0; j < num_columns; ++j){
      temp += (matirx[i * num_columns + j] * input[j] + ROUNDUP) >> 7;
    }
    output[i] = (int16_t) temp;
  }
}

void matrix_vector_multiplication_arm (const int8_t *matrix, int32_t num_rows, int32_t num_columns, const int16_t *input, int16_t *output){
    int8x8_t m;
    int16_t m_lower_part[4];
    int32x4_t temp;
    int16x4_t m_lower;
    for(int32_t i = 0; i < num_rows / 4; ++i){
        temp = vmovq_n_s32(0);
        for(int32_t j = 0; j < num_columns; ++j){
	    m = vld1_s8(matrix + j * num_rows + i);
	    m_lower_part[0] = vdupb_lane_s8(m, 0);
	    m_lower_part[1] = vdupb_lane_s8(m, 1);
	    m_lower_part[2] = vdupb_lane_s8(m, 2);
	    m_lower_part[3] = vdupb_lane_s8(m, 3);
	    m_lower = vld1_s16(m_lower_part);
	    temp = vaddq_s32(temp, vrshrq_n_s32(vmull_n_s16(m_lower, input[j]), 7));
        }
        vst1_s16(output + i * 4, vmovn_s32(temp));
    }
}

//
// Created by zpmc on 2021/12/11.
//


#include <immintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>

#include <unistd.h>
#include <chrono>

#include <x86intrin.h>

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <numeric>
#include <VCL/vectorclass.h>
#include <iostream>

#include <avxintrin.h>
#include <avx2intrin.h>
#include <sse_mathfun.h>

#include "tftn.h"

#define EPS 1e-7
#define FEQU_ZERO(x)  (  -EPS<(x) && (x)<EPS)


#define T256(X) (*((__m256*)&(X)))
#define T128(X) (*((__m128*)&(X)))
using namespace std; //maye it can fix issue https://github.com/ruirangerfan/Three-Filters-to-Normal/issues/4

static inline float FastArcTan(float x)
{
    return M_PI_4*x - x*(fabs(x) - 1)*(0.2447 + 0.0663*fabs(x));
}

inline void atan(Vec8f &x){
    float *p = (float*)(&x);
    *(p++) = atan(*(p));
    *(p++) = atan(*(p));
    *(p++) = atan(*(p));
    *(p++) = atan(*(p));

    *(p++) = atan(*(p));
    *(p++) = atan(*(p));
    *(p++) = atan(*(p));
    *(p) = atan(*(p));
}

inline Vec8f sin(Vec8f &x){
    Vec8f ret;

    *((Vec4f*)&ret) = sin_ps(*((Vec4f*)&x));
    *(((Vec4f*)&ret)+1) = sin_ps(*(((Vec4f*)&x)+1));
    return ret;
}

inline Vec8f cos(Vec8f &x){
    Vec8f ret;
    *((Vec4f*)&ret) = cos_ps(*((Vec4f*)&x));
    *(((Vec4f*)&ret)+1) = cos_ps(*(((Vec4f*)&x)+1));
    return ret;
}


inline void exp_slow(Vec8f &x) {
    *((Vec4f*)&x) = cos_ps(*((Vec4f*)&x));
    *(((Vec4f*)&x)+1) = cos_ps(*(((Vec4f*)&x)+1));
}


inline void exp(Vec8f &x){
    const Vec8f ONE(1.0);
    const Vec8f INV256(1.0/256.0);
    x = ONE + x * INV256;
    x *= x; x *= x; x *= x; x *= x;
    x *= x; x *= x; x *= x; x *= x;
}

#define SURREND(x) idx_m1[-1][x], idx_m1[0][x], idx_m1[1][x],idx[-1][x], idx[1][x],idx_p1[-1][x], idx_p1[0][x], idx_p1[1][x]

static inline void TFTN_MEAN(const Vec8f &kernal_x, const Vec8f &kernal_y,
                             const cv::Mat &input,
                             cv::Mat &output){

    output.create(input.rows,input.cols, CV_32FC3);
    int COL = input.cols;
    cv::Vec3f* idx = (cv::Vec3f*)input.data + COL;
    cv::Vec3f* idx_m1 =(cv::Vec3f*)input.data;
    cv::Vec3f* idx_p1 = (cv::Vec3f*)input.data + COL + COL;
    cv::Vec3f* idx_o = (cv::Vec3f*)output.data + COL;
    cv::Vec3f* end_idx;

    int ONE_MINUS_COL = COL - 1;
    int v(2);
    const Vec8f ZERO(1e5);
    const Vec8f NEG_INF(-1e5);
    const float mul[9]={0,1,1/2.0,1/3.0,1/4.0,1/5.0,1/6.0,1/7.0,1/8.0};
#define UPDATE_IDX idx++, idx_m1++, idx_p1++, idx_o++
    for (; v!=input.rows; ++v){
        end_idx = idx + ONE_MINUS_COL;
        UPDATE_IDX;
        for (; idx != end_idx;UPDATE_IDX){
            Vec8f D = Vec8f(SURREND(2));
            float& nx = idx_o->operator()(0)=horizontal_add(kernal_x / D);
            float& ny = idx_o->operator()(1)=horizontal_add(kernal_y / D);

            if ((*((int*)&nx) & 0x7FFFFFFF) < 1e-7 &&(*((int*)&ny) & 0x7FFFFFFF) < 1e-7) {
                idx_o->operator()(2) = -1;
                continue;
            }
            Vec8f DX =  (idx->operator()(0) - Vec8f(SURREND(0))) * nx;
            Vec8f DY =  (idx->operator()(1) - Vec8f(SURREND(1))) * ny;

            float &x = idx->operator()(0);
            float &y = idx->operator()(1);
            Vec8f Z = D - idx->operator()(2);
            DX = (DX + DY) / Z;
            Vec8fb LT_ZERO = _mm256_cmp_ps(DX, ZERO, _CMP_LE_OS);
            Vec8fb LG_NG_INF = _mm256_cmp_ps(NEG_INF, DX, _CMP_LT_OS);
            Vec8fb BOOL_RESULT = _mm256_and_ps(LT_ZERO, LG_NG_INF);//LT_ZERO & LG_NG_INF;
            Vec8i number=_mm256_srli_epi32(_mm256_castps_si256(BOOL_RESULT), 31);
            float sum = horizontal_add((Vec8f)(_mm256_and_ps(T256(BOOL_RESULT), T256(DX))));
            auto n = horizontal_add(number);
            idx_o->operator()(2) = sum * mul[n];
        }
        UPDATE_IDX;
    }
}

static inline void TFTN_MEDIAN_FAST(const Vec8f &kernal_x, const Vec8f &kernal_y,
                                    const cv::Matx33d &camera,
                                    const cv::Mat &input,
                                    cv::Mat &output){

    output.create(input.rows,input.cols, CV_32FC3);
    int COL = input.cols;
    cv::Vec3f* idx = (cv::Vec3f*)input.data + COL;
    cv::Vec3f* idx_m1 =(cv::Vec3f*)input.data;
    cv::Vec3f* idx_p1 = (cv::Vec3f*)input.data + COL + COL;
    cv::Vec3f* idx_o = (cv::Vec3f*)output.data + COL;
    cv::Vec3f* end_idx;

    int ONE_MINUS_COL = COL - 1;
    int v(2);
    const Vec8f ZERO(1e5);
    const Vec8f NEG_INF(-1e5);
    const float mul[9]={0,1,1/2.0,1/3.0,1/4.0,1/5.0,1/6.0,1/7.0,1/8.0};
#define UPDATE_IDX idx++, idx_m1++, idx_p1++, idx_o++
    for (; v!=input.rows; ++v){
        end_idx = idx + ONE_MINUS_COL;
        UPDATE_IDX;
        for (; idx != end_idx;UPDATE_IDX){
            Vec8f D = Vec8f(SURREND(2));
            float& nx = idx_o->operator()(0)=horizontal_add(kernal_x*camera(0,0) / D);
            float& ny = idx_o->operator()(1)=horizontal_add(kernal_y*camera(1,1) / D);

            if ((*((int*)&nx) & 0x7FFFFFFF) < 1e-7 &&(*((int*)&ny) & 0x7FFFFFFF) < 1e-7) {
                idx_o->operator()(2) = -1;
                continue;
            }
            Vec8f DX =  (idx->operator()(0) - Vec8f(SURREND(0))) * nx;
            Vec8f DY =  (idx->operator()(1) - Vec8f(SURREND(1))) * ny;

            float &x = idx->operator()(0);
            float &y = idx->operator()(1);
            Vec8f Z = D - idx->operator()(2);
            DX = (DX + DY) / Z;
            float *tmp=((float*)(&DX));
            int c=0;
            for (int i = 7; i >= 0; isnan(tmp[i])? i--: tmp[c++] = tmp[i--]);
            std::sort(tmp, tmp+c);
            if (c)  idx_o->operator()(2) = (c&1) ? tmp[c>>1] : (tmp[c>>1] + tmp[(c>>1)-1]) * 0.5;
            else idx_o->operator()(2) = 0;
            continue;
        }
        UPDATE_IDX;
    }
}

static inline void TFTN_MEDIAN_STABLE(const Vec8f &kernal_x, const Vec8f &kernal_y,
                                      const cv::Matx33d &camera,
                                      const cv::Mat &input,
                                      cv::Mat &output){

    output.create(input.rows,input.cols, CV_32FC3);
    int COL = input.cols;
    cv::Vec3f* idx = (cv::Vec3f*)input.data + COL;
    cv::Vec3f* idx_m1 =(cv::Vec3f*)input.data;
    cv::Vec3f* idx_p1 = (cv::Vec3f*)input.data + COL + COL;
    cv::Vec3f* idx_o = (cv::Vec3f*)output.data + COL;
    cv::Vec3f* end_idx;

    int ONE_MINUS_COL = COL - 1;
    int v(2);
    const Vec8f ZERO(1e5);
    const Vec8f NEG_INF(-1e5);
    const float mul[9]={0,1,1/2.0,1/3.0,1/4.0,1/5.0,1/6.0,1/7.0,1/8.0};
#define UPDATE_IDX idx++, idx_m1++, idx_p1++, idx_o++
    for (; v!=input.rows; ++v){
        end_idx = idx + ONE_MINUS_COL;
        UPDATE_IDX;
        for (; idx != end_idx;UPDATE_IDX){
            Vec8f D = Vec8f(SURREND(2));
            float& nx = idx_o->operator()(0)=horizontal_add(kernal_x*camera(0,0) / D);
            float& ny = idx_o->operator()(1)=horizontal_add(kernal_y*camera(1,1) / D);

            if ((*((int*)&nx) & 0x7FFFFFFF) < 1e-7 &&(*((int*)&ny) & 0x7FFFFFFF) < 1e-7) {
                idx_o->operator()(2) = -1;
                continue;
            }
            Vec8f DX =  (idx->operator()(0) - Vec8f(SURREND(0))) * nx;
            Vec8f DY =  (idx->operator()(1) - Vec8f(SURREND(1))) * ny;

            float &x = idx->operator()(0);
            float &y = idx->operator()(1);
            Vec8f Z = D - idx->operator()(2);
            DX = (DX + DY) / Z;
            float *tmp=((float*)(&DX));
            int c=0;
            for (int i = 7; i >= 0; isnan(tmp[i])? i--: tmp[c++] = tmp[i--]);
            std::sort(tmp, tmp+c);
            if (c)  idx_o->operator()(2) = (c&1) ? tmp[c>>1] : (tmp[c>>1] + tmp[(c>>1)-1]) * 0.5;
            else idx_o->operator()(2) = 0;
            continue;
        }
        UPDATE_IDX;
    }
}
#undef SURREND



#define SURREND(x) idx_m1[0][x], idx[-1][x], idx[1][x],idx_p1[0][x]
static inline void TFTN_MEAN(const Vec4f &kernal_x, const Vec4f &kernal_y,
                             const cv::Matx33d &camera,
                             const cv::Mat &input,
                             cv::Mat &output){

    output.create(input.rows,input.cols, CV_32FC3);
    int COL = input.cols;
    cv::Vec3f* idx = (cv::Vec3f*)input.data + COL;
    cv::Vec3f* idx_m1 =(cv::Vec3f*)input.data;
    cv::Vec3f* idx_p1 = (cv::Vec3f*)input.data + COL + COL;
    cv::Vec3f* idx_o = (cv::Vec3f*)output.data + COL;
    cv::Vec3f* end_idx;

    int ONE_MINUS_COL = COL - 1;
    int v(2);
    const Vec4f ZERO(1e5);
    const Vec4f NEG_INF(-1e5);
    const float mul[5]={0,1,1/2.0,1/3.0,1/4.0};
#define UPDATE_IDX idx++, idx_m1++, idx_p1++, idx_o++
    for (; v!=input.rows; ++v){
        end_idx = idx + ONE_MINUS_COL;
        UPDATE_IDX;
        for (; idx != end_idx;UPDATE_IDX){
            Vec4f D = Vec4f(SURREND(2));
            float& nx = idx_o->operator()(0)=horizontal_add(kernal_x*camera(0,0) / D);
            float& ny = idx_o->operator()(1)=horizontal_add(kernal_y*camera(1,1) / D);

            if ((*((int*)&nx) & 0x7FFFFFFF) < 1e-7 &&(*((int*)&ny) & 0x7FFFFFFF) < 1e-7) {
                idx_o->operator()(2) = -1;
                continue;
            }
            Vec4f DX =  (idx->operator()(0) - Vec4f(SURREND(0))) * nx;
            Vec4f DY =  (idx->operator()(1) - Vec4f(SURREND(1))) * ny;

            float &x = idx->operator()(0);
            float &y = idx->operator()(1);
            Vec4f Z = D - idx->operator()(2);
            DX = (DX + DY) / Z;

            Vec4fb LT_ZERO = _mm_cmp_ps(DX, ZERO, _CMP_LE_OS);
            Vec4fb LG_NG_INF = _mm_cmp_ps(NEG_INF, DX, _CMP_LT_OS);
            Vec4fb BOOL_RESULT = _mm_and_ps(LT_ZERO, LG_NG_INF);//LT_ZERO & LG_NG_INF;
            Vec4i number=_mm_srli_epi32(_mm_castps_si128(BOOL_RESULT), 31);
            float sum = horizontal_add((Vec4f)(_mm_and_ps(T128(BOOL_RESULT), T128(DX))));
            auto n = horizontal_add(number);
            idx_o->operator()(2) = sum * mul[n];
        }
        UPDATE_IDX;
    }
}




static inline void TFTN_MEDIAN(const Vec4f &kernal_x, const Vec4f &kernal_y,
                               const cv::Matx33d &camera,
                               const cv::Mat &input,
                               cv::Mat &output){

    output.create(input.rows,input.cols, CV_32FC3);
    int COL = input.cols;
    cv::Vec3f* idx = (cv::Vec3f*)input.data + COL;
    cv::Vec3f* idx_m1 =(cv::Vec3f*)input.data;
    cv::Vec3f* idx_p1 = (cv::Vec3f*)input.data + COL + COL;
    cv::Vec3f* idx_o = (cv::Vec3f*)output.data + COL;
    cv::Vec3f* end_idx;

    int ONE_MINUS_COL = COL - 1;
    int v(2);
    const Vec4f ZERO(1e5);
    const Vec4f NEG_INF(-1e5);
    const float mul[5]={0,1,1/2.0,1/3.0,1/4.0};
#define UPDATE_IDX idx++, idx_m1++, idx_p1++, idx_o++
    for (; v!=input.rows; ++v){
        end_idx = idx + ONE_MINUS_COL;
        UPDATE_IDX;
        for (; idx != end_idx;UPDATE_IDX){
            Vec4f D = Vec4f(SURREND(2));
            float& nx = idx_o->operator()(0)=horizontal_add(kernal_x*camera(0,0) / D);
            float& ny = idx_o->operator()(1)=horizontal_add(kernal_y*camera(1,1) / D);

            if ((*((int*)&nx) & 0x7FFFFFFF) < 1e-7 &&(*((int*)&ny) & 0x7FFFFFFF) < 1e-7) {
                idx_o->operator()(2) = -1;
                continue;
            }
            Vec4f DX =  (idx->operator()(0) - Vec4f(SURREND(0))) * nx;
            Vec4f DY =  (idx->operator()(1) - Vec4f(SURREND(1))) * ny;

            float &x = idx->operator()(0);
            float &y = idx->operator()(1);
            Vec4f Z = D - idx->operator[](2);
            DX = (DX + DY) / Z;

            float *tmp=((float*)(&DX));
            int c=0;
            for (int i = 3; i >= 0; isnan(tmp[i])? i--: tmp[c++] = tmp[i--]);
            std::sort(tmp, tmp+c);
            if (c)  idx_o->operator()(2) = (c&1) ? tmp[c>>1] : (tmp[c>>1] + tmp[(c>>1)-1]) * 0.5;
            else idx_o->operator()(2) = 0;
            continue;
        }
        UPDATE_IDX;
    }
}

void TFTN(const cv::Mat &range_image,
                        const cv::Matx33d camera,
                        const TFTN_METHOD method,
                        cv::Mat* output) {
    const Vec8f kernel_x(-1, 0, 1, -2, 2, -1, 0, 1);
    const Vec8f kernel_y(-1, -2, -1, 0, 0, 1, 2, 1);
    const Vec4f kernel_x4(0, -1, 1, 0);
    const Vec4f kernel_y4(-1, 0, 0, 1);
    const Vec8f kernel_x48(0, 0, 0, -1, 1, 0, 0, 0);
    const Vec8f kernel_y48(0, -1, 0, 0, 0, 0, 1, 0);


    const Vec8f kernel_sobel_x(-1, 0, 1, -2, 2, -1, 0, 1);
    const Vec8f kernel_sobel_y(-1, -2, -1, 0, 0, 1, 2, 1);
    const Vec8f kernel_scharr_x(-3, 0, 3, -10, 10, -3, 0, 3);
    const Vec8f kernel_scharr_y(-3, -10, -3, 0, 0, 3, 10, 3);
    const Vec8f kernel_prewitt_x(-1, 0, 1, -1, 1, -1, 0, 1);
    const Vec8f kernel_prewitt_y(-1, -1, -1, 0, 0, 1, 1, 1);

    switch (method){
        case R_MEANS_8 :
            TFTN_MEAN(-kernel_x * camera(0,0), -kernel_y*camera(1,1), range_image, *output);
            break;
        case R_MEDIAN_FAST_8 :
            TFTN_MEDIAN_FAST(-kernel_x, -kernel_y, camera, range_image, *output);
            break;
        case R_MEDIAN_STABLE_8 :
            TFTN_MEDIAN_STABLE(-kernel_x, -kernel_y, camera, range_image, *output);
            break;
        case R_MEANS_4:
            TFTN_MEAN(-kernel_x4, -kernel_y4, camera, range_image, *output);
            break;
        case R_MEDIAN_4:
            TFTN_MEDIAN(-kernel_x4, -kernel_y4, camera, range_image, *output);
            break;
        case R_MEDIAN_FAST_4_8:
            TFTN_MEDIAN_FAST(-kernel_x48, -kernel_y48, camera, range_image, *output);
            break;
        case R_MEDIAN_STABLE_4_8:
            TFTN_MEDIAN_STABLE(-kernel_x48, -kernel_y48, camera, range_image, *output);
            break;
        case R_MEANS_4_8:
            TFTN_MEAN(-kernel_x48*camera(0,0), -kernel_y48*camera(1,1), range_image, *output);
            break;
        case R_MEANS_SOBEL:
            TFTN_MEAN(-kernel_sobel_x*camera(0,0), -kernel_sobel_y*camera(1,1), range_image, *output);
            break;
        case R_MEDIAN_SOBEL:
            TFTN_MEDIAN_STABLE(-kernel_sobel_x, -kernel_sobel_y, camera, range_image, *output);
            break;
        case R_MEANS_SCHARR:
            TFTN_MEAN(-kernel_scharr_x*camera(0,0), -kernel_scharr_y*camera(1,1), range_image, *output);
            break;
        case R_MEDIAN_SCHARR:
            TFTN_MEDIAN_STABLE(-kernel_scharr_x, -kernel_scharr_y, camera, range_image, *output);
            break;
        case R_MEANS_PREWITT:
            TFTN_MEAN(-kernel_prewitt_x*camera(0,0), -kernel_prewitt_y*camera(1,1), range_image, *output);
            break;
        case R_MEDIAN_PREWITT:
            TFTN_MEDIAN_STABLE(-kernel_prewitt_x, -kernel_prewitt_y, camera, range_image, *output);
            break;
        default:
            std::cerr<<"something wrong?" << std::endl;
            exit(-1);
    }
}



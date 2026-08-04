#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static inline float hsum8(__m256 a) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a,1));
    s = _mm_add_ps(s, _mm_movehl_ps(s,s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}
static inline int hsum_i32_8(__m256i a) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a,1));
    __m128i h = _mm_unpackhi_epi64(s,s);
    s = _mm_add_epi32(s,h);
    h = _mm_shuffle_epi32(s,_MM_SHUFFLE(2,3,0,1));
    return _mm_cvtsi128_si32(_mm_add_epi32(s,h));
}
static inline __m256i unpack_mul(const uint8_t *src) {
    const __m128i il=_mm_setr_epi8(0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3);
    const __m128i ih=_mm_setr_epi8(4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7);
    const __m256i mul=_mm256_setr_epi16(64,16,4,1,64,16,4,1,64,16,4,1,64,16,4,1);
    const __m256i three=_mm256_set1_epi16(3);
    __m128i s=_mm_loadl_epi64((const __m128i*)src);
    __m256i rep=_mm256_set_m128i(_mm_shuffle_epi8(s,ih),_mm_shuffle_epi8(s,il));
    __m256i r0=_mm256_cvtepu8_epi16(_mm256_castsi256_si128(rep));
    __m256i r1=_mm256_cvtepu8_epi16(_mm256_extracti128_si256(rep,1));
    r0=_mm256_and_si256(_mm256_srli_epi16(_mm256_mullo_epi16(r0,mul),6),three);
    r1=_mm256_and_si256(_mm256_srli_epi16(_mm256_mullo_epi16(r1,mul),6),three);
    return _mm256_permute4x64_epi64(_mm256_packus_epi16(r0,r1),0xD8);
}
static inline __m256i unpack_lut(const uint8_t *src) {
    const __m128i lut = _mm_setr_epi8(
        0x00,0x01,0x02,0x03,0x10,0x11,0x12,0x13,
        0x20,0x21,0x22,0x23,0x30,0x31,0x32,0x33);
    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i s = _mm_loadl_epi64((const __m128i *)src);
    const __m128i lo_idx = _mm_and_si128(s, mask);
    const __m128i hi_idx = _mm_and_si128(_mm_srli_epi16(s, 4), mask);
    const __m128i lo_pair = _mm_shuffle_epi8(lut, lo_idx);
    const __m128i hi_pair = _mm_shuffle_epi8(lut, hi_idx);
    const __m128i pairs = _mm_unpacklo_epi8(lo_pair, hi_pair);
    const __m128i c02 = _mm_and_si128(pairs, mask);
    const __m128i c13 = _mm_and_si128(_mm_srli_epi16(pairs,4), mask);
    const __m128i out0 = _mm_unpacklo_epi8(c02,c13);
    const __m128i out1 = _mm_unpackhi_epi8(c02,c13);
    return _mm256_set_m128i(out1,out0);
}
static int scalar32(const uint8_t *x, const int8_t *y) {
    int z=0;
    for(int b=0;b<8;b++) for(int t=0;t<4;t++) z+=(((x[b]>>(2*t))&3)-1)*y[b*4+t];
    return z;
}
static float baseline(const uint8_t*x,const int8_t*y,const float*dx,const float*dy,int nb){
    const __m256i ones=_mm256_set1_epi8(1); float sumf=0;
    for(int i=0;i<nb;i++){float sumi=0;for(int k=0;k<4;k++){
        __m256i c=unpack_mul(x+i*32+k*8),q=_mm256_loadu_si256((const __m256i*)(y+i*128+k*32));
        __m256i dp=_mm256_dpbusd_epi32(_mm256_setzero_si256(),c,q),sy=_mm256_dpbusd_epi32(_mm256_setzero_si256(),ones,q);
        sumi+=dy[i*4+k]*(float)hsum_i32_8(_mm256_sub_epi32(dp,sy));
    }sumf+=dx[i]*sumi;}return sumf;
}
static float global_mul(const uint8_t*x,const int8_t*y,const float*dx,const float*dy,int nb){
    const __m256i ones=_mm256_set1_epi8(1); __m256 acc=_mm256_setzero_ps();
    for(int i=0;i<nb;i++){__m256 block=_mm256_setzero_ps();for(int k=0;k<4;k++){
        __m256i c=unpack_mul(x+i*32+k*8),q=_mm256_loadu_si256((const __m256i*)(y+i*128+k*32));
        __m256i dp=_mm256_dpbusd_epi32(_mm256_setzero_si256(),c,q),sy=_mm256_dpbusd_epi32(_mm256_setzero_si256(),ones,q);
        block=_mm256_fmadd_ps(_mm256_set1_ps(dy[i*4+k]),_mm256_cvtepi32_ps(_mm256_sub_epi32(dp,sy)),block);
    }acc=_mm256_fmadd_ps(_mm256_set1_ps(dx[i]),block,acc);}return hsum8(acc);
}
static float global_lut(const uint8_t*x,const int8_t*y,const float*dx,const float*dy,int nb){
    const __m256i ones=_mm256_set1_epi8(1); __m256 acc=_mm256_setzero_ps();
    for(int i=0;i<nb;i++){__m256 block=_mm256_setzero_ps();for(int k=0;k<4;k++){
        __m256i c=unpack_lut(x+i*32+k*8),q=_mm256_loadu_si256((const __m256i*)(y+i*128+k*32));
        __m256i dp=_mm256_dpbusd_epi32(_mm256_setzero_si256(),c,q),sy=_mm256_dpbusd_epi32(_mm256_setzero_si256(),ones,q);
        block=_mm256_fmadd_ps(_mm256_set1_ps(dy[i*4+k]),_mm256_cvtepi32_ps(_mm256_sub_epi32(dp,sy)),block);
    }acc=_mm256_fmadd_ps(_mm256_set1_ps(dx[i]),block,acc);}return hsum8(acc);
}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec+t.tv_nsec*1e-9;}
typedef float(*fn)(const uint8_t*,const int8_t*,const float*,const float*,int);
static double run(fn f,const uint8_t*x,const int8_t*y,const float*dx,const float*dy,int nb,int r,volatile float*s){double t=now();for(int i=0;i<r;i++)*s+=f(x,y,dx,dy,nb);return now()-t;}
static int cmp(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return(x>y)-(x<y);}
int main(){enum{NB=128,R=20000,T=9};uint8_t*x=aligned_alloc(64,NB*32);int8_t*y=aligned_alloc(64,NB*128);float*dx=aligned_alloc(64,NB*sizeof(float)),*dy=aligned_alloc(64,NB*4*sizeof(float));srand(33);for(int i=0;i<NB*32;i++)x[i]=rand();for(int i=0;i<NB*128;i++)y[i]=(rand()%255)-127;for(int i=0;i<NB;i++){dx[i]=.001f+(rand()%1000)/1000.f;for(int k=0;k<4;k++)dy[i*4+k]=.001f+(rand()%1000)/1000.f;}
for(int i=0;i<NB*4;i++){int a=scalar32(x+i*8,y+i*32);__m256i q=_mm256_loadu_si256((const __m256i*)(y+i*32)),o=_mm256_set1_epi8(1);__m256i cm=unpack_mul(x+i*8),cl=unpack_lut(x+i*8);int b=hsum_i32_8(_mm256_sub_epi32(_mm256_dpbusd_epi32(_mm256_setzero_si256(),cm,q),_mm256_dpbusd_epi32(_mm256_setzero_si256(),o,q)));int c=hsum_i32_8(_mm256_sub_epi32(_mm256_dpbusd_epi32(_mm256_setzero_si256(),cl,q),_mm256_dpbusd_epi32(_mm256_setzero_si256(),o,q)));if(a!=b||a!=c){fprintf(stderr,"decode mismatch %d %d %d at %d\n",a,b,c,i);return 3;}}
float a=baseline(x,y,dx,dy,NB),b=global_mul(x,y,dx,dy,NB),c=global_lut(x,y,dx,dy,NB);printf("rel_mul %.9g rel_lut %.9g\n",fabsf(a-b)/(fabsf(a)+1e-9f),fabsf(a-c)/(fabsf(a)+1e-9f));if(fabsf(a-b)>fmaxf(.05f,fabsf(a)*2e-5f)||fabsf(a-c)>fmaxf(.05f,fabsf(a)*2e-5f))return 4;volatile float s=0;double tb[T],tm[T],tl[T];for(int k=0;k<T;k++){if(k%3==0){tb[k]=run(baseline,x,y,dx,dy,NB,R,&s);tm[k]=run(global_mul,x,y,dx,dy,NB,R,&s);tl[k]=run(global_lut,x,y,dx,dy,NB,R,&s);}else if(k%3==1){tm[k]=run(global_mul,x,y,dx,dy,NB,R,&s);tl[k]=run(global_lut,x,y,dx,dy,NB,R,&s);tb[k]=run(baseline,x,y,dx,dy,NB,R,&s);}else{tl[k]=run(global_lut,x,y,dx,dy,NB,R,&s);tb[k]=run(baseline,x,y,dx,dy,NB,R,&s);tm[k]=run(global_mul,x,y,dx,dy,NB,R,&s);}}qsort(tb,T,sizeof(double),cmp);qsort(tm,T,sizeof(double),cmp);qsort(tl,T,sizeof(double),cmp);double B=tb[T/2],M=tm[T/2],L=tl[T/2];printf("correctness PASS\nbaseline_us %.3f\nglobal_mul_us %.3f speedup %.3fx\nglobal_lut_us %.3f speedup %.3fx vs_mul %.3fx\nsink %g\n",B*1e6/R,M*1e6/R,B/M,L*1e6/R,B/L,M/L,s);return 0;}

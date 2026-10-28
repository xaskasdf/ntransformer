#include "ntrpu_rmsnorm_fp32_r1.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace nt::ntrpu::rmsnorm_r1;
#define CHECK(x) do {if (!(x)) throw std::runtime_error(std::string("line ")+std::to_string(__LINE__)+": " #x);} while(false)
using Vector=std::array<uint32_t,MAX_ELEMENTS>;
static unsigned cases=0,successful=0,values=0;
static uint32_t bits(float f) {uint32_t b;std::memcpy(&b,&f,4);return b;}
static float value(uint32_t b) {float f;std::memcpy(&f,&b,4);return f;}
static uint32_t ordered(uint32_t b) {return b&0x80000000U ? ~b:b|0x80000000U;}
static uint32_t distance(uint32_t a,uint32_t b) {
    const uint32_t x=ordered(a),y=ordered(b);return x>y ? x-y:y-x;
}
static uint64_t fnv(uint64_t hash,uint32_t b) {
    for(unsigned j=0;j<4;++j) hash=(hash^((b>>(8*j))&255U))*0x100000001b3ULL;
    return hash;
}
static void run(const Vector& x,const Vector& w,uint32_t n,uint32_t eps,uint32_t expected,
                const std::string& label,const Vector* fixture=nullptr) {
    std::array<uint32_t,MAX_ELEMENTS+2> guarded;
    guarded.fill(0xa55a5aa5U);const auto original=guarded;
    const auto x_before=x,w_before=w;
    const uint32_t status=ntrpu_rmsnorm_fp32_r1(x.data(),w.data(),guarded.data()+1,n,eps);
    CHECK(status==expected && x==x_before && w==w_before);
    CHECK(guarded.front()==original.front() && guarded.back()==original.back());
    uint32_t max_ulp=0;uint64_t hash=0xcbf29ce484222325ULL;
    if(status!=OK) {
        CHECK(guarded==original);
    } else {
        double sum=0;
        for(uint32_t i=0;i<n;++i) {const double v=value(x[i]);const double square=v*v;sum=sum+square;}
        const double inverse=1.0/std::sqrt(sum/double(n)+double(value(eps)));
        for(uint32_t i=0;i<n;++i) {
            // Independent native conversions/libm, never DUT IEEE helpers.
            const double scaled=double(value(x[i]))*inverse;
            const uint32_t ref=bits(float(scaled*double(value(w[i]))));
            const uint32_t got=guarded[i+1];CHECK(std::isfinite(value(got)));
            const uint32_t ulp=distance(got,ref);CHECK(ulp<=1);
            if((ref&0x7fffffffU)==0) CHECK(got==ref);
            if(fixture) CHECK(distance(got,(*fixture)[i])<=1 && distance(ref,(*fixture)[i])<=1);
            max_ulp=std::max(max_ulp,ulp);hash=fnv(hash,got);
            std::cout<<"NTRPU_RMSNORM_VALUE case="<<cases+1<<" index="<<i<<" bits="
                <<std::hex<<std::setw(8)<<std::setfill('0')<<got<<" ref="<<std::setw(8)<<ref
                <<std::dec<<" ulp="<<ulp<<'\n';
        }
        for(uint32_t i=n;i<MAX_ELEMENTS;++i) CHECK(guarded[i+1]==original[i+1]);
        ++successful;values+=n;
    }
    ++cases;
    std::cout<<"NTRPU_RMSNORM_CASE id="<<cases<<" label="<<label<<" n="<<n
        <<" status="<<status<<" max_ulp="<<max_ulp<<" fnv="<<std::hex<<std::setw(16)
        <<std::setfill('0')<<hash<<std::dec<<'\n';
}
static uint32_t read_u32(std::istream& f) {
    unsigned char b[4];f.read(reinterpret_cast<char*>(b),4);CHECK(bool(f));
    return uint32_t(b[0])|(uint32_t(b[1])<<8)|(uint32_t(b[2])<<16)|(uint32_t(b[3])<<24);
}
int main(int argc,char** argv) {
    try {
        CHECK(argc==2);
        Vector x,w;const uint32_t eps=bits(1e-5f);
        const auto fill=[&]() {for(unsigned i=0;i<MAX_ELEMENTS;++i) {
            x[i]=bits(float(int(i%47)-23)/16);w[i]=bits(float(int(i%19)-9)/8);
        }};
        fill();for(uint32_t n:{1U,31U,32U,33U,256U,2048U}) run(x,w,n,eps,OK,"shape");
        for(unsigned i=0;i<MAX_ELEMENTS;++i) x[i]=i&1 ? 0x80000000U:0;
        run(x,w,2048,eps,OK,"signed_zero");
        for(unsigned i=0;i<MAX_ELEMENTS;++i) {x[i]=(i&1 ? 0x80000000U:0)|uint32_t(i+1);w[i]=bits(1);}
        run(x,w,2048,1,OK,"subnormal_input_epsilon");
        for(unsigned i=0;i<MAX_ELEMENTS;++i) {x[i]=(i&1 ? 0xff7fffffU:0x7f7fffffU);w[i]=bits(.5f);}
        run(x,w,2048,eps,OK,"finite_extremes");
        fill();x[2047]=0x7fc00001U;w[2047]=0xff800000U;run(x,w,31,eps,OK,"outside_extent");
        fill();run(x,w,0,eps,INVALID_SHAPE,"empty");run(x,w,2049,eps,INVALID_SHAPE,"too_large");
        for(uint32_t e:{0U,0x80000000U,0xbf800000U,0x7f800000U,0xff800000U,0x7fc00001U})
            run(x,w,2048,e,INVALID_EPSILON,"bad_epsilon");
        for(uint32_t bad:{0x7fc00001U,0x7f800000U,0xff800000U}) {
            fill();x[2047]=bad;run(x,w,2048,eps,NONFINITE_INPUT,"late_input");
            fill();w[2047]=bad;run(x,w,2048,eps,NONFINITE_INPUT,"late_weight");
        }
        x.fill(0);w.fill(bits(1));x[2047]=bits(1);w[2047]=0x7f7fffffU;
        run(x,w,2048,eps,OUTPUT_OVERFLOW,"late_overflow");
        x[0]=0x7fc00001U;run(x,w,2048,eps,NONFINITE_INPUT,"validation_precedence");
        fill();run(x,w,2048,eps,OK,"recovery");CHECK(cases==27 && successful==11);
        std::ifstream file(argv[1],std::ios::binary);CHECK(bool(file));
        char magic[8];file.read(magic,8);CHECK(std::memcmp(magic,"NTRNORM1",8)==0);
        CHECK(read_u32(file)==33);
        for(unsigned job=0;job<33;++job) {
            CHECK(read_u32(file)==job+1);CHECK(read_u32(file)==2048);
            const uint32_t epsilon=read_u32(file);CHECK(epsilon==eps);
            Vector expected;
            for(auto& v:x) v=read_u32(file);
            for(auto& v:w) v=read_u32(file);
            for(auto& v:expected) v=read_u32(file);
            run(x,w,2048,epsilon,OK,"llama_norm_"+std::to_string(job+1),&expected);
        }
        CHECK(file.peek()==std::char_traits<char>::eof());CHECK(cases==60 && successful==44);
        std::cout<<"NTRPU_RMSNORM_PASS cases="<<cases<<" successful="<<successful
            <<" values="<<values<<" real_norms=33\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}

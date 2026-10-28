#include "ntrpu_residual_fp32_r1.h"
#include <array>
#include <cfenv>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

// Native-only independent IEEE hardware addition; never part of RTL cosim.
static uint32_t next(uint32_t& s) { s^=s<<13; s^=s>>17; s^=s<<5; return s; }
static uint32_t add(uint32_t a,uint32_t b) {
    float x,y; std::memcpy(&x,&a,4); std::memcpy(&y,&b,4);
    volatile float vx=x,vy=y; float z=vx+vy;
    uint32_t bits; std::memcpy(&bits,&z,4); return bits;
}
int main() {
    try {
        if(std::fesetround(FE_TONEAREST)!=0 || add(1,1)!=2) throw std::runtime_error("RNE/gradual underflow unavailable");
        std::array<uint32_t,2048> a{},b{},out{},expected{};
        uint32_t seed=0x82f161b9U; uint64_t values=0,overflows=0;
        for(uint32_t batch=0;batch<1024;++batch) {
            bool overflow=false;
            for(unsigned i=0;i<2048;++i) {
                a[i]=next(seed); b[i]=next(seed);
                if((a[i]&0x7f800000U)==0x7f800000U) a[i]^=0x00800000U;
                if((b[i]&0x7f800000U)==0x7f800000U) b[i]^=0x00800000U;
                // Alternate raw exponent spread and difficult close magnitudes.
                if(batch%4==1) b[i]=(a[i]^0x80000000U)+(next(seed)%3)-1;
                if(batch%4==2) { a[i]&=0x807fffffU; b[i]&=0x807fffffU; }
                if((b[i]&0x7f800000U)==0x7f800000U) b[i]^=0x00800000U;
                expected[i]=add(a[i],b[i]);
                overflow|=(expected[i]&0x7f800000U)==0x7f800000U;
                out[i]=0xa55a5aa5U;
            }
            const auto status=ntrpu_residual_fp32_r1(a.data(),b.data(),out.data(),2048);
            if(status!=(overflow?3U:0U)) throw std::runtime_error("Random status mismatch");
            for(unsigned i=0;i<2048;++i) {
                if(out[i]!=(overflow?0xa55a5aa5U:expected[i])) throw std::runtime_error("Random output mismatch batch="+std::to_string(batch)+" index="+std::to_string(i));
            }
            if(overflow) ++overflows; else values+=2048;
        }
        std::cout<<"NTRPU_RESIDUAL_FUZZ_PASS transactions=1024 input_pairs=2097152 compared_values="
            <<values<<" overflow_transactions="<<overflows<<'\n';
        return 0;
    } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}

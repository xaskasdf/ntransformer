#include "ntrpu_swiglu_q8_block.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace nt::ntrpu::swiglu_block;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string("line ") + \
    std::to_string(__LINE__) + ": " #x); } while (false)

static uint64_t bits(double value) {
    uint64_t result; std::memcpy(&result, &value, 8); return result;
}
static double value(uint64_t pattern) {
    double result; std::memcpy(&result, &pattern, 8); return result;
}
static unsigned cases=0, successes=0;

static void run(const std::array<uint64_t,32>& gate, const std::array<uint64_t,32>& up,
                uint32_t count, uint32_t expected) {
    std::array<uint64_t,10> guarded;
    guarded.fill(0xa55a0123deadbeefULL);
    const auto original=guarded;
    const uint32_t status=ntrpu_swiglu_q8_block(gate.data(),up.data(),guarded.data()+1,count);
    CHECK(status==expected);
    CHECK(guarded[0]==original[0] && guarded[9]==original[9]);
    if (status!=OK) {
        CHECK(guarded==original);
    } else {
        // Independent native IEEE casts and libm, followed by an independent
        // block quantizer; no DUT conversion/quantization helper is used.
        std::array<double,32> reference{};
        double maximum=0;
        for (uint32_t i=0;i<count;++i) {
            const double g=float(value(gate[i])),u=float(value(up[i]));
            const double e=std::exp(-std::abs(g));
            reference[i]=float((g*(g>=0 ? 1/(1+e) : e/(1+e)))*u);
            maximum=std::max(maximum,std::abs(reference[i]));
        }
        const double scale=maximum/127.0;
        const double actual_scale=value(guarded[5]);
        CHECK(std::isfinite(actual_scale) && actual_scale>=0);
        CHECK(std::abs(actual_scale-scale)<=std::max(std::numeric_limits<double>::denorm_min(),2e-7*scale));
        for (uint32_t i=0;i<32;++i) {
            const int raw=int((guarded[1+i/8]>>((i%8)*8))&255U);
            const int got=raw<128 ? raw:raw-256;
            const int want=scale==0 ? 0:int(std::round(reference[i]/scale));
            CHECK(got==want);
            if (i>=count) CHECK(got==0);
        }
        CHECK(guarded[6]==0 && guarded[7]==0 && guarded[8]==0);
        successes++;
    }
    cases++;
    std::cout<<"NTRPU_SWIGLU_BLOCK_CASE id="<<cases<<" elements="<<count<<" status="<<status;
    if (status==OK) for(unsigned i=0;i<8;++i)
        std::cout<<" word"<<i<<"="<<std::hex<<std::setw(16)<<std::setfill('0')<<guarded[i+1]<<std::dec;
    std::cout<<'\n';
}

int main() {
    try {
        std::array<uint64_t,32> g,u;
        for(unsigned i=0;i<32;++i) {g[i]=bits((int(i)-17)/4.0);u[i]=bits((int(i%9)-4)/3.0);}
        for(uint32_t count:{1U,16U,31U,32U}) run(g,u,count,OK);
        g.fill(bits(0));u.fill(bits(1));run(g,u,32,OK);
        g.fill(bits(-0.0));run(g,u,32,OK);
        const double tiny=std::numeric_limits<float>::denorm_min();
        for(unsigned i=0;i<32;++i) {g[i]=bits((i&1 ? -1:1)*tiny*(i+1));u[i]=bits(std::numeric_limits<float>::max());}
        run(g,u,32,OK);
        g.fill(bits(257));u.fill(bits(1));run(g,u,32,OK);
        g.fill(bits(-257));u.fill(bits(std::numeric_limits<float>::max()));run(g,u,32,OK);
        g.fill(bits(1));u.fill(bits(1));
        g[31]=bits(std::numeric_limits<double>::quiet_NaN());
        u[31]=bits(std::numeric_limits<double>::infinity());run(g,u,31,OK);
        g.fill(bits(1));u.fill(bits(1));
        run(g,u,0,INVALID_SHAPE);run(g,u,33,INVALID_SHAPE);
        for(uint64_t bad:{0x7ff8000000000000ULL,0x7ff0000000000000ULL,0xfff0000000000000ULL}) {
            g[31]=bad;run(g,u,32,NONFINITE_INPUT);g[31]=bits(1);
            u[31]=bad;run(g,u,32,NONFINITE_INPUT);u[31]=bits(1);
        }
        g[31]=bits(std::nextafter(double(std::numeric_limits<float>::max()),std::numeric_limits<double>::infinity()));
        run(g,u,32,NARROW_OVERFLOW);g[31]=bits(1);
        u[31]=bits(-std::numeric_limits<double>::max());run(g,u,32,NARROW_OVERFLOW);u[31]=bits(1);
        g[31]=u[31]=bits(std::numeric_limits<float>::max());run(g,u,32,ACTIVATION_OVERFLOW);
        g[0]=bits(std::numeric_limits<double>::infinity());run(g,u,32,NONFINITE_INPUT);
        g[0]=bits(std::numeric_limits<double>::max());g[31]=bits(std::numeric_limits<double>::quiet_NaN());
        run(g,u,32,NONFINITE_INPUT);
        // Recovery plus FP64->FP32 midpoint rounding and gradual underflow.
        for(unsigned i=0;i<32;++i) {
            const double midpoint=(double(1.0f)+double(std::nextafter(1.0f,2.0f)))/2;
            g[i]=bits(i%3==0 ? midpoint : i%3==1 ? std::nextafter(midpoint,0.0):std::nextafter(midpoint,2.0));
            u[i]=bits((int(i)-16)/7.0);
        }
        run(g,u,32,OK);
        g.fill(bits(tiny/2));u.fill(bits(1));run(g,u,32,OK);
        g.fill(bits(0));u.fill(bits(1));run(g,u,1,OK);
        // Valid INT8 payload words can have the bit pattern of a binary64 NaN.
        // This must remain a raw ABI record; a floating-point result writer
        // must not reject or canonicalize its first four words.
        g.fill(bits(257));u.fill(bits(-1));u[0]=bits(127);run(g,u,32,OK);
        CHECK(cases==27 && successes==14);
        std::cout<<"PASS SwiGLU Q8 block cases="<<cases<<" successful="<<successes<<'\n';
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;
    }
}

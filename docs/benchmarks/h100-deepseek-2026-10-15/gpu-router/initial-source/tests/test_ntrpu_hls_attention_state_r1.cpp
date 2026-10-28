#include "ntrpu_attention_tile_fp32_r1.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

using namespace nt::ntrpu::attention_tile_r1;
static uint32_t bits(float x) { uint32_t v; std::memcpy(&v, &x, 4); return v; }
static uint64_t calls = 0, failures = 0, outputs = 0;
static void need(bool condition, const char* why) {
    if (!condition) throw std::runtime_error(why);
}
struct Bank {
    std::array<uint32_t,64> q{};
    std::array<uint32_t,4096> k{}, v{};
    std::array<uint64_t,104> state{};
    std::array<uint64_t,106> next{};
    std::array<uint32_t,66> out{};
    uint32_t position=0, layer=0, head=0, kv=0;
    uint64_t sequence=0xfedcba9876543210ULL, epoch=0x1234567887654321ULL;
    uint32_t call(uint32_t start, uint32_t count, uint32_t expected=OK) {
        const auto oldq=q;
        const auto oldk=k, oldv=v;
        const auto olds=state;
        next.fill(0xdeadbeef00badbadULL); out.fill(0xa5a55a5aU);
        const auto status=ntrpu_attention_tile_fp32_r1(q.data(),k.data(),v.data(),state.data(),next.data()+1,out.data()+1,
            start,count,position,layer,head,kv,sequence,epoch);
        ++calls;
        if (status!=expected) {
            std::printf("FAIL: call=%llu start=%u count=%u position=%u status=%u expected=%u\n",
                static_cast<unsigned long long>(calls),start,count,position,status,expected);
            throw std::runtime_error("Unexpected status");
        }
        need(q==oldq && k==oldk && v==oldv && state==olds,"Input modified");
        need(next.front()==0xdeadbeef00badbadULL && next.back()==0xdeadbeef00badbadULL,"State guard modified");
        need(out.front()==0xa5a55a5aU && out.back()==0xa5a55a5aU,"Output guard modified");
        if (status!=OK) {
            ++failures;
            need(std::all_of(next.begin(),next.end(),[](uint64_t x){return x==0xdeadbeef00badbadULL;}),"Error published state");
        } else {
            need(next[1]==STATE_MAGIC && next[2]==sequence && next[3]==epoch && next[4]==position &&
                 next[5]==((uint64_t(layer)<<32)|head) && next[6]==uint64_t(start)+count,"State identity");
            std::copy(next.begin()+1,next.end()-1,state.begin());
        }
        if (status!=OK || uint64_t(start)+count!=uint64_t(position)+1)
            need(std::all_of(out.begin(),out.end(),[](uint32_t x){return x==0xa5a55a5aU;}),"Partial/error published output");
        else outputs+=64;
        return status;
    }
};

static void singleton() {
    for (uint32_t h=0;h<32;++h) {
        Bank b;b.layer=h%16;b.head=h;b.kv=h/4;
        const uint32_t data[]={0,0x80000000U,1,0x80000001U,0x007fffffU,0x00800000U,
            0x7f7fffffU,0xff7fffffU,0x3f800000U,0xbf800000U};
        for (uint32_t d=0;d<64;++d) {b.q[d]=data[(d+2)%10];b.k[d]=data[(d+4)%10];b.v[d]=data[d%10];}
        // Fresh jobs ignore all state bytes and inactive tile positions.
        b.state.fill(0xffffffffffffffffULL);b.k.back()=0x7fc00001U;b.v.back()=0xff800000U;
        b.call(0,1);
        for (uint32_t d=0;d<64;++d) need(b.out[d+1]==b.v[d],"One-token identity");
    }
}
static void error_cases() {
    Bank initial;initial.position=3;initial.q[0]=bits(1);initial.v.fill(bits(2));initial.call(0,1);
    for (uint32_t index: {0U,1U,2U,3U,4U,5U,8U,39U}) {
        Bank b=initial;b.state[index]^=1;b.call(1,1,INVALID_STATE);
    }
    for (uint32_t index: {6U,7U,40U,103U}) {
        Bank b=initial;b.state[index]=0x7ff8000000000001ULL;b.call(1,1,INVALID_STATE);
    }
    for (uint64_t bad: {0ULL,0x3fe0000000000000ULL,0x4000000000000000ULL}) {
        Bank b=initial;b.state[7]=bad;b.call(1,1,INVALID_STATE);
    }
    {Bank b=initial;b.state[6]=0x7fefffffffffffffULL;b.call(1,1,INVALID_STATE);}
    {Bank b=initial;b.state[103]=0x7fefffffffffffffULL;b.call(1,1,INVALID_STATE);}
    {Bank b=initial;b.q[63]^=1;b.call(1,1,INVALID_STATE);}
    {Bank b=initial;b.epoch^=1;b.call(1,1,INVALID_STATE);}
    {Bank b=initial;b.sequence^=1;b.call(1,1,INVALID_STATE);}
    {Bank b=initial;b.position=4;b.call(1,1,INVALID_STATE);}
    {Bank b=initial;b.head=1;b.call(1,1,INVALID_STATE);}
    for (uint32_t count: {0U,65U,0xffffffffU}) {Bank b=initial;b.call(0,count,INVALID_SHAPE);}
    {Bank b=initial;b.call(0xffffffffU,1,INVALID_SHAPE);}
    {Bank b=initial;b.position=131072;b.call(0,1,INVALID_SHAPE);}
    {Bank b=initial;b.layer=16;b.call(0,1,INVALID_IDENTITY);}
    {Bank b=initial;b.head=32;b.call(0,1,INVALID_IDENTITY);}
    {Bank b=initial;b.kv=1;b.call(0,1,INVALID_IDENTITY);}
    {Bank b=initial;b.epoch=0;b.call(0,1,INVALID_IDENTITY);}
    {Bank b=initial;b.sequence=0;b.call(0,1,INVALID_IDENTITY);}
    for (uint32_t poison: {0x7f800000U,0xff800000U,0x7fc00001U}) {
        {Bank b=initial;b.q[63]=poison;b.call(0,1,NONFINITE_INPUT);}
        {Bank b=initial;b.k[255]=poison;b.call(0,4,NONFINITE_INPUT);}
        {Bank b=initial;b.v[255]=poison;b.call(0,4,NONFINITE_INPUT);}
    }
    // State errors precede nonfinite inputs on continuation; shape precedes both.
    {Bank b=initial;b.state[0]=0;b.q[63]=0x7fc00001U;b.call(1,1,INVALID_STATE);b.call(1,0,INVALID_SHAPE);}
    initial.call(1,3);
    for (uint32_t d=0;d<64;++d) need(initial.out[d+1]==bits(2),"Recovery after rejected jobs");
}
static void extreme_dot() {
    for (uint32_t qword: {0x7f7fffffU,0xff7fffffU}) {
        Bank b;b.position=2;b.q.fill(qword);b.k.fill(0x7f7fffffU);b.v.fill(bits(1));
        b.call(0,1);b.call(1,1);b.call(2,1);
        for (uint32_t d=0;d<64;++d) need(b.out[d+1]==bits(1),"Extreme-dot uniform values");
    }
}
static void equal_values() {
    for (uint32_t value: {0x7f7fffffU,0xff7fffffU,1U,0x80000001U,bits(1)}) {
        for (uint32_t n: {2U,3U,7U,31U,63U,64U,65U,127U,129U,1025U}) {
            Bank b;b.q[0]=bits(8);b.position=n-1;b.v.fill(value);
            for (uint32_t start=0;start<n;) {
                const uint32_t count=std::min(64U,n-start);
                for (uint32_t t=0;t<count;++t) b.k[t*64]=bits(float(int((start+t)*37%193)-96)/16);
                b.call(start,count);start+=count;
            }
            for (uint32_t d=0;d<64;++d) need(b.out[d+1]==value,"Constant-value convex mean");
        }
    }
}
static Bank partition(uint32_t stride, uint32_t n) {
    Bank b;b.position=n-1;
    for (uint32_t d=0;d<64;++d)b.q[d]=bits(float(int(d%17)-8)/16);
    for (uint32_t start=0;start<n;) {
        uint32_t count=std::min(stride,n-start);
        for (uint32_t t=0;t<count;++t) for (uint32_t d=0;d<64;++d) {
            b.k[t*64+d]=bits(float(int(((start+t)*73+d*19)%257)-128)/32);
            b.v[t*64+d]=bits(float(int(((start+t)*41+d*13)%509)-254)/16);
        }
        b.call(start,count);start+=count;
    }
    return b;
}
static void partitions() {
    for (uint32_t n: {65U,127U,129U,1025U}) {
        const auto ref=partition(64,n);
        for (uint32_t stride: {1U,3U,7U,31U,63U}) {
            const auto got=partition(stride,n);
            need(got.state==ref.state && got.out==ref.out,"Tile partition changed operation order");
        }
    }
}
static void full_context() {
    Bank b;b.position=MAX_CONTEXT-1;b.q[0]=bits(1);b.k.fill(0);b.v.fill(bits(1));
    for (uint32_t start=0;start<MAX_CONTEXT;start+=64)b.call(start,64);
    for (uint32_t d=0;d<64;++d)need(b.out[d+1]==bits(1),"Full-context uniform result");
    need(b.state[5]==MAX_CONTEXT,"Full context not scanned");
}
int main() {
    try {
        singleton();error_cases();extreme_dot();equal_values();partitions();full_context();
        std::printf("NTRPU_ATTENTION_STATE_PASS calls=%llu rejected=%llu output_words=%llu full_context=131072\n",
            static_cast<unsigned long long>(calls),static_cast<unsigned long long>(failures),static_cast<unsigned long long>(outputs));
        return 0;
    } catch (const std::exception& error) {std::printf("FAIL: %s\n",error.what());return 1;}
}

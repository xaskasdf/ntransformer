#include "ntrpu_norm_q8_abi1_r1.h"
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace nt::ntrpu::norm_q8_r1;
#define CHECK(x) do {if (!(x)) throw std::runtime_error(std::string("line ")+std::to_string(__LINE__)+": " #x);} while(false)
static uint32_t read32(std::istream& f) {
    unsigned char b[4]; f.read(reinterpret_cast<char*>(b),4); CHECK(bool(f));
    return uint32_t(b[0])|(uint32_t(b[1])<<8)|(uint32_t(b[2])<<16)|(uint32_t(b[3])<<24);
}
static uint64_t read64(std::istream& f) { const uint32_t low=read32(f),high=read32(f); return low|(uint64_t(high)<<32); }
int main(int argc,char** argv) {
    try {
        CHECK(argc==2); std::ifstream file(argv[1],std::ios::binary); CHECK(bool(file));
        char magic[8]; file.read(magic,8); CHECK(std::memcmp(magic,"NTRNQ8R1",8)==0);
        const uint32_t cases=read32(file); CHECK(cases>0 && cases<256);
        uint32_t successful=0,words=0,records=0;
        for(uint32_t id=1;id<=cases;++id) {
            CHECK(read32(file)==id);
            const uint32_t elements=read32(file),epsilon=read32(file),want_status=read32(file),n=read32(file);
            CHECK(want_status<=5 && n<=MAX_WORDS);
            CHECK((want_status==OK && elements>0 && elements<=2048 && elements%32==0 && n==elements/4) ||
                  (want_status!=OK && n==0));
            std::array<uint32_t,MAX_ELEMENTS> input,weight;
            std::array<uint64_t,MAX_WORDS> expected;
            for(auto& v:input) v=read32(file);
            for(auto& v:weight) v=read32(file);
            for(uint32_t i=0;i<n;++i) expected[i]=read64(file);
            const auto old_input=input,old_weight=weight;
            std::array<uint64_t,MAX_WORDS+2> guarded; guarded.fill(0xa55a5aa53cc3c33cULL);
            const auto before=guarded;
            const uint32_t status=ntrpu_norm_q8_abi1_r1(input.data(),weight.data(),guarded.data()+1,elements,epsilon);
            CHECK(status==want_status && input==old_input && weight==old_weight);
            CHECK(guarded.front()==before.front() && guarded.back()==before.back());
            uint64_t hash=0xcbf29ce484222325ULL;
            if(status==OK) {
                for(uint32_t i=0;i<n;++i) {
                    const uint64_t got=guarded[i+1]; CHECK(got==expected[i]);
                    if(i%8>=5) CHECK(got==0);
                    if(i%8==4) CHECK(got<=0x4780204070e1c387ULL);
                    for(unsigned j=0;j<8;++j) hash=(hash^((got>>(8*j))&255U))*0x100000001b3ULL;
                    std::cout<<"NTRPU_NORM_Q8_WORD case="<<id<<" index="<<i<<" bits="
                        <<std::hex<<std::setw(16)<<std::setfill('0')<<got<<std::dec<<'\n';
                }
                for(uint32_t i=n;i<MAX_WORDS;++i) CHECK(guarded[i+1]==before[i+1]);
                ++successful; words+=n; records+=elements/32;
            } else CHECK(guarded==before);
            std::cout<<"NTRPU_NORM_Q8_CASE id="<<id<<" elements="<<elements<<" epsilon="
                <<std::hex<<std::setw(8)<<std::setfill('0')<<epsilon<<std::dec<<" status="<<status<<" words="<<n
                <<" fnv="<<std::hex<<std::setw(16)<<hash<<std::dec<<'\n';
        }
        CHECK(file.peek()==std::char_traits<char>::eof());
        std::cout<<"NTRPU_NORM_Q8_PASS cases="<<cases<<" successful="<<successful<<" words="<<words<<" records="<<records<<'\n';
        return 0;
    } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}

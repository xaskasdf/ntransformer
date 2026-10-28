#include "ntrpu_residual_fp32_r1.h"
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace nt::ntrpu::residual_r1;
#define CHECK(x) do {if (!(x)) throw std::runtime_error(std::string("line ")+std::to_string(__LINE__)+": " #x);} while(false)
static uint32_t read_word(std::istream& f) {
    unsigned char b[4]; f.read(reinterpret_cast<char*>(b),4); CHECK(bool(f));
    return uint32_t(b[0])|(uint32_t(b[1])<<8)|(uint32_t(b[2])<<16)|(uint32_t(b[3])<<24);
}
int main(int argc,char** argv) {
    try {
        CHECK(argc==2); std::ifstream file(argv[1],std::ios::binary); CHECK(bool(file));
        char magic[8]; file.read(magic,8); CHECK(std::memcmp(magic,"NTRRADD1",8)==0);
        const uint32_t cases=read_word(file); CHECK(cases>0 && cases<256);
        uint32_t successful=0,values=0;
        for(uint32_t id=1;id<=cases;++id) {
            CHECK(read_word(file)==id);
            const uint32_t elements=read_word(file),expected_status=read_word(file),n=read_word(file);
            CHECK(expected_status<=3 && n<=MAX_ELEMENTS);
            CHECK((expected_status==OK && elements>0 && elements<=MAX_ELEMENTS && n==elements) ||
                  (expected_status!=OK && n==0));
            std::array<uint32_t,MAX_ELEMENTS> residual,branch,expected;
            for(auto& v:residual) v=read_word(file);
            for(auto& v:branch) v=read_word(file);
            for(uint32_t i=0;i<n;++i) expected[i]=read_word(file);
            const auto old_residual=residual,old_branch=branch;
            std::array<uint32_t,MAX_ELEMENTS+2> guarded; guarded.fill(0xa55a5aa5U);
            const auto before=guarded;
            const uint32_t status=ntrpu_residual_fp32_r1(residual.data(),branch.data(),guarded.data()+1,elements);
            CHECK(status==expected_status && residual==old_residual && branch==old_branch);
            CHECK(guarded.front()==before.front() && guarded.back()==before.back());
            uint64_t hash=0xcbf29ce484222325ULL;
            if(status==OK) {
                for(uint32_t i=0;i<n;++i) {
                    const uint32_t got=guarded[i+1]; CHECK(got==expected[i]);
                    for(unsigned j=0;j<4;++j) hash=(hash^((got>>(8*j))&255U))*0x100000001b3ULL;
                    std::cout<<"NTRPU_RESIDUAL_VALUE case="<<id<<" index="<<i<<" bits="
                        <<std::hex<<std::setw(8)<<std::setfill('0')<<got<<std::dec<<'\n';
                }
                for(uint32_t i=n;i<MAX_ELEMENTS;++i) CHECK(guarded[i+1]==before[i+1]);
                ++successful; values+=n;
            } else CHECK(guarded==before);
            std::cout<<"NTRPU_RESIDUAL_CASE id="<<id<<" elements="<<elements<<" status="<<status<<" values="<<n
                <<" fnv="<<std::hex<<std::setw(16)<<std::setfill('0')<<hash<<std::dec<<'\n';
        }
        CHECK(file.peek()==std::char_traits<char>::eof());
        std::cout<<"NTRPU_RESIDUAL_PASS cases="<<cases<<" successful="<<successful<<" values="<<values<<'\n';
        return 0;
    } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}

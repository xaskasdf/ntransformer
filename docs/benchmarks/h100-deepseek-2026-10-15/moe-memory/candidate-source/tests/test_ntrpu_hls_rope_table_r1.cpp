#include "ntrpu_rope_table_fp32_r1.h"
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace nt::ntrpu::rope_table_r1;
#define CHECK(x) do {if (!(x)) throw std::runtime_error(std::string("line ")+std::to_string(__LINE__)+": " #x);} while(false)
static uint32_t read_word(std::istream& f) {
    unsigned char b[4];f.read(reinterpret_cast<char*>(b),4);CHECK(bool(f));
    return uint32_t(b[0])|(uint32_t(b[1])<<8)|(uint32_t(b[2])<<16)|(uint32_t(b[3])<<24);
}
static float value(uint32_t b) {float f;std::memcpy(&f,&b,4);return f;}
static uint32_t bits(float f) {uint32_t b;std::memcpy(&b,&f,4);return b;}
static uint64_t fnv(uint64_t hash,uint32_t b) {
    for(unsigned j=0;j<4;++j) hash=(hash^((b>>(8*j))&255U))*0x100000001b3ULL;
    return hash;
}
int main(int argc,char** argv) {
    try {
        CHECK(argc==2);std::ifstream file(argv[1],std::ios::binary);CHECK(bool(file));
        char magic[8];file.read(magic,8);CHECK(std::memcmp(magic,"NTRROPE1",8)==0);
        const uint32_t cases=read_word(file);CHECK(cases>0 && cases<128);
        uint32_t successful=0,values=0;
        for(uint32_t id=1;id<=cases;++id) {
            CHECK(read_word(file)==id);
            const uint32_t heads=read_word(file),position=read_word(file),phase_position=read_word(file);
            const uint32_t expected_status=read_word(file),elements=read_word(file);
            CHECK(expected_status<=5 && elements<=MAX_ELEMENTS);
            CHECK((expected_status==OK && heads>0 && heads<=MAX_HEADS && elements==heads*HEAD_DIM) ||
                  (expected_status!=OK && elements==0));
            std::array<uint32_t,64> coefficients;
            std::array<uint32_t,MAX_ELEMENTS> input,expected;
            for(auto& v:coefficients) v=read_word(file);
            for(auto& v:input) v=read_word(file);
            for(uint32_t i=0;i<elements;++i) expected[i]=read_word(file);
            const auto input_before=input;const auto coeff_before=coefficients;
            std::array<uint32_t,MAX_ELEMENTS+2> guarded;guarded.fill(0xa55a5aa5U);
            const auto before=guarded;
            const uint32_t status=ntrpu_rope_table_fp32_r1(input.data(),coefficients.data(),guarded.data()+1,
                                                        heads,position,phase_position);
            CHECK(status==expected_status && input==input_before && coefficients==coeff_before);
            CHECK(guarded.front()==before.front() && guarded.back()==before.back());
            uint64_t hash=0xcbf29ce484222325ULL;
            if(status==OK) {
                for(uint32_t i=0;i<elements;i+=2) {
                    const double x=value(input[i]),y=value(input[i+1]);
                    const double c=value(coefficients[i%64]),s=value(coefficients[i%64+1]);
                    const double xc=x*c,ys=y*s,xs=x*s,yc=y*c;
                    const uint32_t refs[2]={bits(float(xc-ys)),bits(float(xs+yc))};
                    for(unsigned k=0;k<2;++k) {
                        const uint32_t got=guarded[i+k+1];CHECK(std::isfinite(value(got)));
                        CHECK(got==refs[k] && got==expected[i+k]);
                        hash=fnv(hash,got);
                        std::cout<<"NTRPU_ROPE_VALUE case="<<id<<" index="<<i+k<<" bits="
                            <<std::hex<<std::setw(8)<<std::setfill('0')<<got<<std::dec<<'\n';
                    }
                }
                for(uint32_t i=elements;i<MAX_ELEMENTS;++i) CHECK(guarded[i+1]==before[i+1]);
                ++successful;values+=elements;
            } else CHECK(guarded==before);
            std::cout<<"NTRPU_ROPE_CASE id="<<id<<" heads="<<heads<<" position="<<position
                <<" coefficient_position="<<phase_position<<" status="<<status<<" values="<<elements
                <<" fnv="<<std::hex<<std::setw(16)<<std::setfill('0')<<hash<<std::dec<<'\n';
        }
        CHECK(file.peek()==std::char_traits<char>::eof());
        std::cout<<"NTRPU_ROPE_PASS cases="<<cases<<" successful="<<successful<<" values="<<values<<'\n';
        return 0;
    } catch(const std::exception& e) {std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}

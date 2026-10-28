#include "ntrpu_attention_tile_fp32_r1.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

static void need(bool x,const char* why) {if(!x)throw std::runtime_error(why);}
template<class T> static void read(std::ifstream& f,T* data,size_t n) {
    f.read(reinterpret_cast<char*>(data),sizeof(T)*n);need(bool(f),"Truncated fixture");
}
static uint32_t order(uint32_t x) {return (x&0x80000000U)?~x:(x|0x80000000U);}
static uint32_t distance(uint32_t a,uint32_t b) {
    if((a&0x7fffffffU)==0&&(b&0x7fffffffU)==0)return 0;
    const auto x=order(a),y=order(b);return x>y?x-y:y-x;
}
int main(int argc,char** argv) {
    try {
        need(argc==2,"Expected fixture path");std::ifstream f(argv[1],std::ios::binary);need(bool(f),"Cannot open fixture");
        char magic[8];read(f,magic,8);need(std::memcmp(magic,"NTRATTN1",8)==0,"Fixture magic");
        uint32_t cases;read(f,&cases,1);need(cases==143,"Fixture cases");
        uint32_t transactions=0,words=0,max_ulp=0;
        for(uint32_t ident=1;ident<=cases;++ident) {
            uint32_t h[6];uint64_t tags[2];read(f,h,6);read(f,tags,2);
            need(h[0]==ident && h[1]>=1 && h[1]<=129 && h[5]>=1 && h[5]<=64,"Fixture header");
            std::array<uint32_t,64> q,expected;read(f,q.data(),64);
            std::vector<uint32_t> allk(h[1]*64),allv(h[1]*64);read(f,allk.data(),allk.size());read(f,allv.data(),allv.size());read(f,expected.data(),64);
            std::array<uint64_t,104> state{};
            for(uint32_t start=0;start<h[1];) {
                uint32_t count=std::min(h[5],h[1]-start);
                std::array<uint32_t,4096> k{},v{};
                std::copy_n(allk.data()+start*64,count*64,k.data());std::copy_n(allv.data()+start*64,count*64,v.data());
                std::array<uint64_t,106> next;next.fill(0xdeadbeef00badbadULL);
                std::array<uint32_t,66> out;out.fill(0xa5a55a5aU);
                const auto oldq=q;
                const auto oldk=k,oldv=v;
                const auto olds=state;
                auto status=ntrpu_attention_tile_fp32_r1(q.data(),k.data(),v.data(),state.data(),next.data()+1,out.data()+1,
                    start,count,h[1]-1,h[2],h[3],h[4],tags[0],tags[1]);
                need(status==0,"Attention status");need(q==oldq&&k==oldk&&v==oldv&&state==olds,"Input modified");
                need(next.front()==0xdeadbeef00badbadULL&&next.back()==0xdeadbeef00badbadULL,"State canaries");
                need(out.front()==0xa5a55a5aU&&out.back()==0xa5a55a5aU,"Output canaries");
                std::printf("NTRPU_ATTENTION_TILE case=%u start=%u count=%u status=%u\n",ident,start,count,status);
                for(uint32_t i=0;i<104;++i) {
                    state[i]=next[i+1];std::printf("NTRPU_ATTENTION_STATE case=%u start=%u index=%u bits=%016llx\n",
                        ident,start,i,static_cast<unsigned long long>(state[i]));
                }
                if(start+count==h[1]) {
                    for(uint32_t d=0;d<64;++d) {
                        const auto delta=distance(out[d+1],expected[d]);max_ulp=std::max(max_ulp,delta);
                        need(delta<=1,"Output exceeds one ULP from offline reference");
                        std::printf("NTRPU_ATTENTION_WORD case=%u index=%u bits=%08x\n",ident,d,out[d+1]);++words;
                    }
                } else need(std::all_of(out.begin(),out.end(),[](uint32_t x){return x==0xa5a55a5aU;}),"Partial published output");
                start+=count;++transactions;
            }
        }
        need(f.peek()==std::ifstream::traits_type::eof(),"Fixture trailing bytes");
        std::printf("NTRPU_ATTENTION_PASS cases=%u transactions=%u words=%u offline_max_ulp=%u\n",cases,transactions,words,max_ulp);
        return 0;
    } catch(const std::exception& e) {std::printf("FAIL: %s\n",e.what());return 1;}
}

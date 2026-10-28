// One token, all 16 layers and tied vocabulary output. Native kernel source only.
// Expected files are comparison targets; every layer consumes native prior output.
#include "ntrpu_norm_q8_abi1_r1.h"
#include "ntrpu_rmsnorm_fp32_r1.h"
#include "ntrpu_q8_block_mac.h"
#include "ntrpu_rope_table_fp32_r1.h"
#include "ntrpu_attention_tile_fp32_r1.h"
#include "ntrpu_quantize_abi1_r1.h"
#include "ntrpu_post_attention_r1.h"
#include "ntrpu_swiglu.h"
#include "ntrpu_swiglu_q8_block.h"
#include "ntrpu_residual_fp32_r1.h"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t D=2048,H=8192,V=128256;
constexpr uint32_t EPS=0x3727c5ac;
using Bytes=std::vector<uint8_t>;
using Words=std::vector<uint32_t>;
using Abi=std::vector<uint64_t>;
static_assert(std::endian::native==std::endian::little);
std::filesystem::path reference,output;
std::ifstream model;
std::map<std::string,std::pair<uint64_t,uint64_t>> tensor_index;
uint64_t compared=0,mac_blocks=0;
bool inject_tail=false;

void need(bool good,const std::string& why) {if(!good)throw std::runtime_error(why);}
void status(uint32_t code,const std::string& where) {
    if(code)throw std::runtime_error(where+" status="+std::to_string(code));
}
Bytes read_bytes(const std::filesystem::path& path,size_t size) {
    need(std::filesystem::file_size(path)==size,"File size "+path.string());
    Bytes bytes(size);std::ifstream file(path,std::ios::binary);
    need(bool(file.read(reinterpret_cast<char*>(bytes.data()),size)),"Read "+path.string());return bytes;
}
template<class T> std::vector<T> tensor(const std::string& name,size_t count,uint64_t offset=0,bool slice=false) {
    const auto [base,size]=tensor_index.at(name);
    need(offset+count*sizeof(T)<=size && (slice || count*sizeof(T)==size),"Tensor range "+name);
    model.clear();model.seekg(base+offset);std::vector<T> data(count);
    need(bool(model.read(reinterpret_cast<char*>(data.data()),count*sizeof(T))),"Tensor read "+name);return data;
}
template<class T> void check(const std::string& name,const std::vector<T>& data) {
    auto expected=read_bytes(reference/(name+".bin"),data.size()*sizeof(T));
    if(inject_tail && name=="logits-fp32")expected.back()^=1;
    auto actual=reinterpret_cast<const uint8_t*>(data.data());
    for(size_t i=0;i<expected.size();++i)
        if(actual[i]!=expected[i])throw std::runtime_error(name+" byte="+std::to_string(i)+" actual="+
             std::to_string(actual[i])+" expected="+std::to_string(expected[i]));
    compared+=expected.size();std::cout<<"FORWARD_COMPARE name="<<name<<" bytes="<<expected.size()<<" exact=1\n";
}
template<class T> void save(const std::string& name,const std::vector<T>& data) {
    const auto path=output/(name+".bin");need(!std::filesystem::exists(path),"Output exists");
    std::ofstream stream(path,std::ios::binary);
    need(bool(stream.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(T))),"Output write");
}
Words narrow(const std::vector<double>& values) {
    Words words;words.reserve(values.size());
    for(double value:values) {float f=0;need(nt::ntrpu::swiglu_hls::detail::narrow(value,f),"Narrow overflow");words.push_back(std::bit_cast<uint32_t>(f));}
    return words;
}
std::vector<double> project(const Bytes& weight,size_t rows,const Abi& input) {
    size_t blocks=input.size()/8;need(weight.size()==rows*blocks*34,"Projection shape");
    std::vector<double> result(rows,0);
    for(size_t row=0;row<rows;++row)for(size_t b=0;b<blocks;++b) {
        const uint8_t* w=weight.data()+(row*blocks+b)*34;uint16_t scale;
        std::array<uint64_t,4> words{};std::memcpy(&scale,w,2);std::memcpy(words.data(),w+2,32);
        double sx=std::bit_cast<double>(input[b*8+4]),next=0;
        status(ntrpu_q8_block_mac(words[0],words[1],words[2],words[3],input[b*8],input[b*8+1],
            input[b*8+2],input[b*8+3],scale,sx,result[row],&next),"MAC");result[row]=next;++mac_blocks;
    }
    return result;
}
Abi norm(const Words& input,const Words& weights) {
    Abi out(512);status(ntrpu_norm_q8_abi1_r1(input.data(),weights.data(),out.data(),D,EPS),"Norm/Q8");return out;
}
Abi quantize(const Words& input) {
    Abi out(512);status(ntrpu_quantize_abi1_r1(input.data(),out.data(),D),"Quantize");return out;
}
double half_value(uint16_t bits) {
    const unsigned exponent=(bits>>10)&31,mantissa=bits&1023;
    need(exponent!=31,"Nonfinite embedding scale");
    const double value=exponent ? std::ldexp(double(1024+mantissa),int(exponent)-25):std::ldexp(double(mantissa),-24);
    return bits&0x8000 ? -value:value;
}
}

int main(int argc,char** argv) {
    try {
        need(argc==5 || argc==6,"Expected MODEL INDEX REFERENCE NEW_OUTPUT [--inject-last-logit]");
        inject_tail=argc==6;need(!inject_tail || std::string(argv[5])=="--inject-last-logit","Unknown flag");
        model.open(argv[1],std::ios::binary);need(bool(model),"Open model");
        std::ifstream table(argv[2]);std::string name;uint64_t offset,size;
        while(table>>name>>offset>>size)need(tensor_index.emplace(name,std::pair(offset,size)).second,"Duplicate tensor");
        need(table.eof() && tensor_index.size()==147,"Tensor index");
        reference=argv[3];output=argv[4];need(!std::filesystem::exists(output),"Existing output");
        std::filesystem::create_directories(output);
        const auto started=std::chrono::steady_clock::now();
        auto emb=tensor<uint8_t>("token_embd.weight",2176,0,true);Words current(D);
        for(size_t b=0;b<64;++b) {
            uint16_t scale;std::memcpy(&scale,emb.data()+b*34,2);
            for(size_t lane=0;lane<32;++lane) {
                const int8_t q=std::bit_cast<int8_t>(emb[b*34+2+lane]);
                current[b*32+lane]=std::bit_cast<uint32_t>(float(half_value(scale)*q));
            }
        }
        check("embedding-fp32",current);
        std::array<uint32_t,64> phase{};for(size_t i=0;i<64;i+=2)phase[i]=0x3f800000;
        for(uint32_t layer=0;layer<16;++layer) {
            const std::string p="layer-"+(layer<10?std::string("0"):std::string())+std::to_string(layer)+"-";
            const std::string t="blk."+std::to_string(layer)+".";
            check(p+"input-fp32",current);
            auto abi=norm(current,tensor<uint32_t>(t+"attn_norm.weight",D));check(p+"input-abi",abi);
            std::vector<Words> qkv;
            for(auto [part,rows]:std::array<std::pair<const char*,size_t>,3>{{{"q",D},{"k",512},{"v",512}}}) {
                auto full=project(tensor<uint8_t>(t+"attn_"+part+".weight",rows*2176),rows,abi);
                check(p+part+"-fp64",full);qkv.push_back(narrow(full));check(p+part+"-fp32",qkv.back());
            }
            Words qk=qkv[0];qk.insert(qk.end(),qkv[1].begin(),qkv[1].end());Words rotated(2560);
            status(ntrpu_rope_table_fp32_r1(qk.data(),phase.data(),rotated.data(),40,0,0),"RoPE");check(p+"qk-rope-fp32",rotated);
            Words attention(D);std::vector<double> scores;
            for(uint32_t head=0;head<32;++head) {
                std::array<uint32_t,4096> keys{},values{};std::array<uint64_t,104> in{},out{};
                std::copy_n(rotated.data()+D+(head/4)*64,64,keys.data());
                std::copy_n(qkv[2].data()+(head/4)*64,64,values.data());
                status(ntrpu_attention_tile_fp32_r1(rotated.data()+head*64,keys.data(),values.data(),
                    in.data(),out.data(),attention.data()+head*64,0,1,0,layer,head,head/4,1,1),"Attention");
                need(out[0]==nt::ntrpu::attention_tile_r1::STATE_MAGIC && out[1]==1 && out[2]==1 && out[3]==0 &&
                     out[4]==((uint64_t(layer)<<32)|head) && out[5]==1 && std::bit_cast<double>(out[7])==1,"Attention identity");
                scores.push_back(std::bit_cast<double>(out[6]));
            }
            check(p+"attention-score-fp64",scores);check(p+"attention-fp32",attention);
            auto oabi=quantize(attention);check(p+"o-abi",oabi);
            auto ofull=project(tensor<uint8_t>(t+"attn_output.weight",D*2176),D,oabi);check(p+"o-fp64",ofull);
            auto branch=narrow(ofull);check(p+"o-fp32",branch);
            Words post(D);Abi fabi(512);auto fw=tensor<uint32_t>(t+"ffn_norm.weight",D);
            status(ntrpu_post_attention_r1(current.data(),branch.data(),fw.data(),post.data(),fabi.data(),D,EPS),"Post-O");
            check(p+"post-residual-fp32",post);check(p+"ffn-abi",fabi);
            auto gate=project(tensor<uint8_t>(t+"ffn_gate.weight",H*2176),H,fabi);
            auto up=project(tensor<uint8_t>(t+"ffn_up.weight",H*2176),H,fabi);
            check(p+"gate-fp64",gate);check(p+"up-fp64",up);
            Abi actabi(H/4);std::vector<float> act(H);
            for(size_t b=0;b<H/32;++b) {
                std::array<uint64_t,32> gb{},ub{};std::array<float,32> gf{},uf{};
                for(size_t l=0;l<32;++l) {
                    gb[l]=std::bit_cast<uint64_t>(gate[b*32+l]);ub[l]=std::bit_cast<uint64_t>(up[b*32+l]);
                    need(nt::ntrpu::swiglu_hls::detail::narrow(gate[b*32+l],gf[l]),"Gate narrow");
                    need(nt::ntrpu::swiglu_hls::detail::narrow(up[b*32+l],uf[l]),"Up narrow");
                }
                status(ntrpu_swiglu(gf.data(),uf.data(),act.data()+b*32,32),"SwiGLU");
                status(ntrpu_swiglu_q8_block(gb.data(),ub.data(),actabi.data()+b*8,32),"SwiGLU/ABI");
            }
            check(p+"swiglu-fp32",act);check(p+"swiglu-abi",actabi);
            auto down=project(tensor<uint8_t>(t+"ffn_down.weight",D*(H/32)*34),D,actabi);check(p+"down-fp64",down);
            auto rounded=narrow(down);check(p+"down-fp32",rounded);Words result(D);
            status(ntrpu_residual_fp32_r1(post.data(),rounded.data(),result.data(),D),"Final residual");
            check(p+"output-fp32",result);save(p+"output-fp32",result);current=std::move(result);
            std::cout<<"FORWARD_LAYER layer="<<layer<<" complete=1\n"<<std::flush;
        }
        auto abi=norm(current,tensor<uint32_t>("output_norm.weight",D));check("output-abi",abi);
        std::vector<double> logits;logits.reserve(V);
        for(size_t first=0;first<V;first+=8192) {
            size_t rows=std::min(size_t(8192),V-first);
            auto part=project(tensor<uint8_t>("token_embd.weight",rows*2176,first*2176,true),rows,abi);
            logits.insert(logits.end(),part.begin(),part.end());
            std::cout<<"FORWARD_HEAD first="<<first<<" rows="<<rows<<"\n"<<std::flush;
        }
        check("logits-fp64",logits);auto logit_words=narrow(logits);check("logits-fp32",logit_words);save("logits-fp32",logit_words);
        size_t best=0;for(size_t i=1;i<V;++i)if(std::bit_cast<float>(logit_words[i])>std::bit_cast<float>(logit_words[best]))best=i;
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        std::cout<<"FORWARD_PASS layers=16 token=0 position=0 logits=128256 argmax="<<best
                 <<" compared_bytes="<<compared<<" mac_blocks="<<mac_blocks<<" cpu_seconds="<<seconds<<"\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<"FORWARD_FAIL "<<e.what()<<"\n";return 1;}
}

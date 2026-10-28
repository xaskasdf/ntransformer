#include "model/ds41/ds41_forward.h"
#include "model/ds41/ds41_gpu.h"
#include <stdexcept>
#include <string>
#include <cstdio>
#include <fstream>
#include <filesystem>
int main(int argc,char** argv) {
    try {
        if(argc!=3)throw std::runtime_error("model token_id");
        nt::ds41::Model model;std::string error;
        if(!model.load(argv[1],error))throw std::runtime_error(error);
        if (const char* path=std::getenv("GAP_DUMP")) {
            auto dir=std::filesystem::path(path).parent_path()/"operator-weights";
            std::filesystem::create_directories(dir);
            for(const char* name:{"blk.0.attn_q_a.weight","blk.0.attn_q_a_norm.weight"}) {
                auto t=model.tensor(name);std::ofstream out(dir/name,std::ios::binary);
                out.write(reinterpret_cast<const char*>(t.data),t.rows()*t.row_bytes());
            }
            std::ofstream meta(dir/"config.json");
            meta<<"{\"eps\":"<<model.cfg().norm_eps<<",\"rows\":"<<model.cfg().q_lora_rank<<",\"cols\":"<<model.cfg().hidden<<"}\n";
        }
        auto head=model.tensor("output.weight");
        if(!nt::ds41::gpu_lm_head_init(head.data,head.rows(),head.ne0()))throw std::runtime_error("head init failed");
        nt::ds41::Session session(model);const float* logits=session.forward(std::stoi(argv[2]));
        if(const char* path=std::getenv("GAP_DUMP")) {
            std::ofstream out(std::filesystem::path(path)/"first-logits.f32",std::ios::binary);
            out.write(reinterpret_cast<const char*>(logits),model.cfg().vocab_size*sizeof(float));
        }
        nt::ds41::gpu_print_stats();
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}

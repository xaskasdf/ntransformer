# Reference GPU only. The NTRPU library is compiled with CUDA disabled.
FROM pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime@sha256:6acf597eeb8e376a96580dde4952f37cc017fef732bb40bfc73f28f25e3f64b4

RUN apt-get update && apt-get install -y --no-install-recommends g++ make \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir numpy==2.5.3 huggingface_hub==2.0.0 cmake==4.3.1

WORKDIR /work
ENTRYPOINT ["bash", "/workspace/scripts/run_ntrpu_reference.sh"]

"""Reference tensor boundaries taken from the accelerator's arithmetic contract."""
import struct
import torch


def round_bf16_rnd8(value):
    bits=value.float().contiguous().view(torch.int32)
    upper=((bits>>16)+((bits>>15)&1))&0xffff
    exponent=(bits>>23)&0xff
    upper=torch.where(exponent==0,(bits>>16)&0x8000,upper)
    upper=torch.where((exponent==255)&((bits&0x7fffff)!=0),0x7fc0,upper)
    return (upper<<16).view(torch.float32)


def configure_reference(model):
    model.bf=round_bf16_rnd8
    # EPS CSR is BF16 0x3586 expanded to FP32, as in the deployed PS firmware.
    epsilon=struct.unpack('<f',struct.pack('<I',0x35860000))[0]
    def norm(x,gamma):
        mean=torch.mean(x*x,dim=-1,keepdim=True)
        return model.bf(model.bf(x*torch.rsqrt(mean+epsilon))*gamma)
    model.norm=norm
    model.project_bias=lambda x,weight,bias:qkv_projection(model,x,weight,bias)
    return model


def qkv_projection(model,x,weight,bias):
    q,scale=weight
    # GEMM rounds once, then BF16 scale product and FP32 bias add round together.
    # Rounding after scale and again after bias would introduce an extra boundary.
    raw=model.bf(torch.nn.functional.linear(x,q))
    return model.bf(raw*scale+bias)

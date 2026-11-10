// Layout shared by the device kernels (nv2_dev.cu) and the host engine (nv2_host.cpp).
// All structs below live in one pinned, GPU-mapped host region (anon mmap + cudaHostRegister(Mapped)).
#pragma once
#include "penv.h"
#include <cstdint>

namespace nv2s {

constexpr int MAXU = 384;   // unique experts per layer call (E; DeepSeek-V4.1-Flash: 384)
constexpr int MAXP = 64;    // picks with per-pick payload (decode: <= 8 tokens x top-8)
constexpr int RQ = 64;      // request / reply ring entries
constexpr int DFLAG_LAYERS = 127;   // nv_step's per-layer X16 request records: dflag[1 + li] for li < DFLAG_LAYERS
constexpr int LG = 8192;    // write-back / admission log ring entries
constexpr int LR = 256;     // landing ring (fresh RAM slots for VRAM victims)
constexpr int MAXB = 8;     // tokens per CPU job

// host control words (mapped int64)
// HC_CPDONE: seq of the last request whose nv_copy finished; layer-ahead staging DMAs start after it, so they do not
// contend for PCIe with that layer's own admission copies
enum { HC_PUB = 0, HC_ACK = 1, HC_ERR = 2, HC_WBN = 3, HC_ADMN = 4, HC_LANDT = 5, HC_CDONE = 6, HC_CPDONE = 7, HC_MOEDONE = 8, HC_N = 64 };
// device control words (device int64)
enum { DC_SEQ = 0, DC_LANDH = 1, DC_WBN = 2, DC_ADMN = 3, DC_NU = 4, DC_TPUB = 5, DC_CDONE = 6, DC_N = 16 };
constexpr int DBGW = 16;   // debug ring words per request (0-7 pub/step/copy, 8-9 combine start/go, 10 fix done)
constexpr int DBG = 4096;   // latency debug ring (device: tpub, t_step, t_seen; host: t_notice, t_reply)
// lanes: the host's plan for each unique expert of a layer call (Rep.lane[u])
//   LN_VRAM           GPU, VRAM-resident
//   LN_RAM / LN_NVG   GPU, admitted to VRAM by nv_copy from its RAM slot (LN_NVG: once its NVMe read has landed there)
//   LN_STG            GPU, admitted like LN_RAM, but nv_copy reads the expert from its VRAM staging slot (Rep.stg[u])
//   LN_RZC / LN_NZC   GPU, zero-copy from the RAM slot (LN_NZC: once its NVMe read has landed there), not admitted
//   LN_CPU / LN_NVC   CPU lane, masked out of the GPU selection (LN_NVC: read from NVMe)
//   LN_SKIP           lossy verify: cold pick skipped (masked, weight 0), no read
enum { LN_RAM = 0, LN_NVG = 1, LN_CPU = 2, LN_VRAM = 3, LN_NVC = 4, LN_RZC = 5, LN_NZC = 6, LN_SKIP = 8, LN_STG = 9 };
// request kinds
enum { RK_DECODE = 0, RK_SMALL = 1, RK_VERIFY = 2 };   // RK_VERIFY: decode-class call the server flagged as a verify step

struct Req {                 // device -> host, one per MoE layer call (seq written last)
    int64_t seq;
    int64_t tpub;            // device globaltimer at publication (ns)
    int32_t li, nu, np, ntok, kind, pad;
    int32_t key[MAXU];       // unique picked keys (li * E + e), expert id ascending
    int32_t cls[MAXU];       // LN_VRAM if VRAM-resident at publication, else 0
    int32_t cnt[MAXU];       // tokens that picked it
    int32_t pe[MAXP];        // per pick: local expert id (np <= MAXP only)
    float pw[MAXP];          // per pick: routing weight
    int32_t npf, pad2;       // prefetch hints: predicted picks of the NEXT layer (not VRAM-resident), deduped
    int32_t pf_key[MAXP];
    int32_t has_lg, has_plg; // 1: lg holds this layer's router logits / plg the layer-ahead guess for layer li + 1
    uint16_t lg[MAXU];       // fp16 router logits of all E experts (residency policy / trace)
    uint16_t plg[MAXU];      // fp16 predicted router logits of layer li + 1 (prefetch / staging policy, trace)
};

struct Rep {                 // host -> device (seq written last)
    int64_t seq;
    int32_t ncpu, nnv;
    int32_t lane[MAXU];
    int32_t rn_on, rn_topk;  // lossy verify: pick s is scaled by rn_sc[s / rn_topk] (restores the skipped routed mass)
    float rn_sc[MAXB];
    int32_t stg[MAXU];       // LN_STG: staging pool slot holding the expert (device address table nv2 stg_addr)
};

struct LogE { int64_t seq; int32_t key, slot; };

// stats (device uint64) indices
enum { ST_HIT = 0, ST_MISS = 1, ST_ADMIT = 2, ST_NOVICT = 3, ST_EVICT = 4, ST_WB = 5, ST_WB_NOSLOT = 6, ST_FIX = 7,
       ST_BAD = 8, ST_WAITNS = 9, ST_WAITS = 10, ST_CWAITNS = 11, ST_CWAITS = 12, ST_TIMEOUT = 13, ST_N = 16 };

}  // namespace nv2s

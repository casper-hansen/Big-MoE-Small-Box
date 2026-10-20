// N119 nv2 host engine (one per GPU process). Owns every residency decision ("only the host marks an expert
// resident, and only after its bytes have landed"):
//   controller thread  polls the device's per-layer requests (mapped ring), drains the device's write-back /
//                      admission logs (exclusive RAM tier), runs plan_layer (moetier semantics: lanes gpu / zerocopy /
//                      cpu / nvme, greedy min-max split, NVMe stall), issues O_DIRECT NVMe reads into fresh RAM slots,
//                      writes the reply (lanes), dispatches the CPU job, keeps the landing ring + free list stocked
//   reader pool        pthreads doing O_DIRECT preads of record pieces (decode reads ahead of prefill read-ahead);
//                      the thread that finishes a record's last piece marks it landed (ram_res = 1, after the bytes)
//   CPU worker         AVX2 MUL1 kernels (ft_core.h) computing RAM-resident experts IN PLACE from their RAM slot
//                      (native layout), slots pinned (refcount) for the job; persistent pool (no per-layer spin-up)
//   stage engine       prefill: per forward, pins the RAM-resident experts, reads the NVMe-only ones ahead (FIFO ring,
//                      bounded queue depth, several layers ahead) and H2Ds them into the VRAM staging buffers
// Eviction never touches a key of the request being served, a pinned slot, an in-flight read, or a landing slot.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <unordered_map>
#include <algorithm>
#include <execinfo.h>
#include <signal.h>
#include "ft_core.h"
#include "ft_n135.h"
// N135 CPU-lane forward switch: GLM53_NV_CPU_KERN=1 -> n135::moe_forward4 (ft_n135.h), default 0 -> ft_core moe_forward
static int g_cpu_kern = -1;
static inline void cpu_kern_init()
{
    if (g_cpu_kern >= 0) return;
    const char* s = getenv("GLM53_NV_CPU_KERN");
    g_cpu_kern = s && *s ? atoi(s) : 0;
    n135::configure_env(); n135::init_p16();
    fprintf(stderr, "[nv2] CPU lane kernel: %s (rg %d cg %d rd %d cd %d bandmaxm %d pfr %d p16 %d rt %d merge %d vbmi %d)\n", g_cpu_kern == 1 ? "n135 v4" : "ft_core moe_forward",
            n135::K_.rg, n135::K_.cg, n135::K_.rd, n135::K_.cd, n135::K_.bandmaxm, n135::K_.pfr, n135::K_.kern, n135::K_.rt, n135::K_.merge,
            int(n135::vbmi_ok));
}
static inline void cpu_moe_forward(Pool& pool, const Layer& L, const float* x, int ntok, const std::vector<std::vector<std::pair<int, float>>>& route,
                                   float* out, int mode, PhaseTimes* pt = nullptr)
{
    cpu_kern_init();
    if (g_cpu_kern == 1) n135::moe_forward4(pool, L, x, ntok, route, out, mode);
    else moe_forward(pool, L, x, ntok, route, out, mode, pt);
}
#include "nv2_shared.h"
#include <sys/mman.h>
#include <fcntl.h>

using namespace nv2s;
void dev_bind(pybind11::module& m);

namespace nv2h {

// N137 B70 ring layout (b70/ring.py): HDR int64[512] | KEYS int32[16384] | REQ | X fp16[MAXP][H] | OUT fp16[MAXP][H]
constexpr int64_t B70_MAGIC = 0x4E31333752494E47LL;
constexpr int64_t B70_OFF_KEYS = 4096, B70_OFF_REQ = 69632, B70_OFF_X = 73728, B70_OFF_OUT = 73728 + 64LL * 4096 * 2;
constexpr int64_t B70_SIZE = ((B70_OFF_OUT + 64LL * 4096 * 2 + 4095) / 4096) * 4096;
enum { B70_W_MAGIC = 0, B70_W_STATE = 4, B70_W_LOAD_SEQ = 6, B70_W_LOAD_ACK = 7, B70_W_NKEYS = 8, B70_W_REQ_SEQ = 9,
       B70_W_DONE_SEQ = 10, B70_W_ERR = 11, B70_W_NSLOTS = 18, B70_W_HEARTBEAT = 19 };

static inline long long realtime_ns() { struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec; }
static inline double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static inline void cpu_relax() { _mm_pause(); }
static void pin_self(int cpu) { if (cpu < 0) return; cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s); pthread_setaffinity_np(pthread_self(), sizeof(s), &s); }
static void pin_self_set(const std::vector<int>& cpus) { if (cpus.empty()) return; cpu_set_t s; CPU_ZERO(&s); for (int c : cpus) CPU_SET(c, &s); pthread_setaffinity_np(pthread_self(), sizeof(s), &s); }

// ------------------------------------------------------------------------------------------------------------
// reader pool
// ------------------------------------------------------------------------------------------------------------
typedef void (*DoneFn)(void* ctx, int64_t tag, int err, double ms);
// Read classes, strict priority at piece dispatch (an SSD request in flight cannot be preempted, so what bounds a demand
// read's latency is the lower-class bytes already in flight):
//   RC_DEMAND    a layer call waits for it now (GPU copy / fix, CPU lane): FIFO, up to hi_cap bytes in flight, so the
//                records of one layer complete one after another instead of all at the end
//   RC_PREFETCH  layer-ahead hints: newest first (LIFO), only while no demand piece is queued and the total in flight is
//                below lo_cap (the interference a newly arriving demand read can see); promoted to RC_DEMAND when a layer
//                call demands the key while its pieces are still queued
//   RC_BULK      prefill staging, warm start: below demand, not capped (the stage engine bounds its own queue depth)
enum { RC_DEMAND = 0, RC_PREFETCH = 1, RC_BULK = 2, RC_N = 3 };
typedef void (*PartFn)(void* ctx, int64_t tag);
struct Track
{
    std::atomic<int> left{0}; std::atomic<int> err{0}; DoneFn fn = nullptr; void* ctx = nullptr; int64_t tag = 0; double t0 = 0;
    int fd = -1; int64_t off = 0, dst = 0, len = 0, piece = 0, next = 0; int cls = 0; int key = -1;
    std::atomic<uint64_t> done_mask{0}; uint64_t part_mask = 0; PartFn part = nullptr; std::atomic<int> part_fired{0};
};
struct Piece { int fd; int64_t off, dst, len; Track* tr; int cls; double vdone; };

struct Reader
{
    std::vector<std::thread> th;
    std::mutex mu; std::condition_variable cv;
    std::deque<Track*> q[RC_N];
    int64_t infl[RC_N] = { 0, 0, 0 };          // bytes in flight per class
    int64_t hi_cap = int64_t(12) << 20, lo_cap = int64_t(6) << 20;
    bool lifo = true, prio = true;
    std::unordered_map<int, Track*> queued;     // key -> track that still has undispatched pieces (classes > 0)
    bool quit = false;
    std::atomic<long> recs{0}, pieces{0}, promoted{0};
    // device emulation (set_read_rate, diagnostics): pieces complete no faster than a single device of rate_bps would
    // serve them in dispatch order (0 = off)
    double rate_bps = 0, vfin = 0;
    static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
    void start(int n, std::vector<int> cpus)
    {
        if (const char* v = getenv("GLM53_NV_HI_MB")) hi_cap = int64_t(atof(v) * 1048576.0);
        if (const char* v = getenv("GLM53_NV_LO_MB")) lo_cap = int64_t(atof(v) * 1048576.0);
        if (const char* v = getenv("GLM53_NV_PF_LIFO")) lifo = atoi(v) != 0;
        if (const char* v = getenv("GLM53_NV_PRIO")) prio = atoi(v) != 0;
        fprintf(stderr, "[nv2] reader: %d threads, priority %d, demand cap %.1f MB, prefetch cap %.1f MB, prefetch lifo %d\n",
                n, int(prio), hi_cap / 1048576.0, lo_cap / 1048576.0, int(lifo));
        for (int i = 0; i < n; ++i) th.emplace_back([this, cpus] { pin_self_set(cpus); loop(); });
    }
    void submit(int fd, int64_t off, int64_t dst, int64_t len, int64_t piece, int cls, DoneFn fn, void* ctx, int64_t tag, int key = -1,
                PartFn part = nullptr, int64_t part_bytes = 0)
    {
        if (!prio) cls = cls == RC_BULK ? RC_BULK : RC_DEMAND;   // legacy: demand and prefetch share one FIFO
        Track* tr = new Track; tr->fn = fn; tr->ctx = ctx; tr->tag = tag; tr->t0 = now_ms();
        if (part && part_bytes > 0 && (part_bytes + piece - 1) / piece <= 63)
        {
            tr->part = part;
            const int np = int((part_bytes + piece - 1) / piece);
            tr->part_mask = np >= 64 ? ~uint64_t(0) : ((uint64_t(1) << np) - 1);
        }
        tr->fd = fd; tr->off = off; tr->dst = dst; tr->len = len; tr->piece = piece; tr->cls = cls; tr->key = key;
        tr->left = int((len + piece - 1) / piece);
        {
            std::lock_guard<std::mutex> lk(mu);
            q[cls].push_back(tr);
            if (key >= 0 && cls > 0) queued[key] = tr;
        }
        recs++;
        cv.notify_all();
    }
    // a layer call demands key: its queued (lower-class) pieces move to the demand queue
    bool promote(int key)
    {
        std::lock_guard<std::mutex> lk(mu);
        auto it = queued.find(key);
        if (it == queued.end()) return false;
        Track* tr = it->second;
        queued.erase(it);
        if (tr->cls == RC_DEMAND) return false;
        auto& d = q[tr->cls];
        auto f = std::find(d.begin(), d.end(), tr);
        if (f != d.end()) d.erase(f);
        tr->cls = RC_DEMAND;
        q[RC_DEMAND].push_back(tr);
        promoted++;
        cv.notify_all();
        return true;
    }
    int pick() const   // caller holds mu: class allowed to start a piece now, or -1
    {
        if (!q[RC_DEMAND].empty()) return infl[RC_DEMAND] < hi_cap ? RC_DEMAND : -1;
        const int64_t tot = infl[0] + infl[1] + infl[2];
        if (!q[RC_PREFETCH].empty() && tot < lo_cap) return RC_PREFETCH;
        if (!q[RC_BULK].empty()) return RC_BULK;
        return -1;
    }
    void loop()
    {
        for (;;)
        {
            Piece p;
            {
                std::unique_lock<std::mutex> lk(mu);
                int c = -1;
                cv.wait(lk, [&] { return quit || (c = pick()) >= 0; });
                if (quit) return;
                auto& d = q[c];
                Track* tr = (c == RC_PREFETCH && lifo) ? d.back() : d.front();
                const int64_t n = std::min(tr->piece, tr->len - tr->next);
                double vd = 0;
                if (rate_bps > 0) { vfin = std::max(vfin, now_s()) + double(n) / rate_bps; vd = vfin; }
                p = { tr->fd, tr->off + tr->next, tr->dst + tr->next, n, tr, c, vd };
                tr->next += n;
                if (tr->next >= tr->len)
                {
                    if (c == RC_PREFETCH && lifo) d.pop_back(); else d.pop_front();
                    if (tr->key >= 0) { auto it = queued.find(tr->key); if (it != queued.end() && it->second == tr) queued.erase(it); }
                }
                infl[c] += n;
            }
            int64_t done = 0; int err = 0;
            while (done < p.len)
            {
                ssize_t r = pread(p.fd, (void*) (p.dst + done), size_t(p.len - done), off_t(p.off + done));
                if (r < 0) { if (errno == EINTR) continue; err = errno; break; }
                if (r == 0) { err = -1; break; }
                done += r;
            }
            if (p.vdone > 0)
            {
                const double w = p.vdone - now_s();
                if (w > 0) std::this_thread::sleep_for(std::chrono::duration<double>(w));
            }
            {
                std::lock_guard<std::mutex> lk(mu);
                infl[p.cls] -= p.len;
            }
            cv.notify_all();
            pieces++;
            if (err) p.tr->err = err;
            if (p.tr->part && !err)
            {
                const int idx = int((p.off - p.tr->off) / p.tr->piece);
                const uint64_t m = p.tr->done_mask.fetch_or(uint64_t(1) << std::min(idx, 63)) | (uint64_t(1) << std::min(idx, 63));
                if ((m & p.tr->part_mask) == p.tr->part_mask && p.tr->part_fired.exchange(1) == 0) p.tr->part(p.tr->ctx, p.tr->tag);
            }
            if (p.tr->left.fetch_sub(1) == 1)
            {
                Track* tr = p.tr;
                tr->fn(tr->ctx, tr->tag, tr->err.load(), now_ms() - tr->t0);
                delete tr;
            }
        }
    }
    void stop() { { std::lock_guard<std::mutex> lk(mu); quit = true; } cv.notify_all(); for (auto& t : th) t.join(); th.clear(); }
};

// ------------------------------------------------------------------------------------------------------------
// the engine
// ------------------------------------------------------------------------------------------------------------
enum { KS_NONE = 0, KS_INFLIGHT = 1, KS_RES = 2 };
enum { SL_FREE = -1, SL_LAND = -2, SL_STAGE = -3 };

struct Pol
{
    double g0 = 0.17, thit = 0.023, tzc = 0.38, ca = 0.145, cb = 0.165, ctok = 0.040, push = 0.40;
    double nvlat = 0.12, nvdeep = 0.395, nvone = 0.75;
    double tnzc = 0.6;    // GPU cost of a zero-copy NVMe pick after landing, ms (GLM53_NV_TNZC; < 0 = legacy model)
    int cpu_on = 0, nvcpu_on = 0, maxcpu = 32, nv_noadmit = 0;
};

struct Counters
{
    std::atomic<long long> reqs{0}, dec_reqs{0}, small_reqs{0}, picks{0}, vram{0}, ram{0}, nvme{0}, inflight_hit{0},
        lane_zc{0}, lane_cpu{0}, lane_nvg{0}, lane_nvc{0}, reads{0}, read_bytes{0}, wb_landed{0}, wb_dup{0}, adm_freed{0},
        adm_pinned{0}, evictions{0}, cpu_jobs{0}, cpu_experts{0}, cpu_tokens{0}, stage_ram{0}, stage_nv{0},
        stage_reads{0}, read_err{0}, alloc_fail{0}, land_pushed{0}, prefetch{0}, dec_tok{0}, wb_issued{0}, wb_drop{0}, vring_hits{0}, notice_ns{0}, notice_max_ns{0}, serve_max_ns{0};
    std::atomic<long long> lane_b70{0}, b70_jobs{0}, b70_rows{0}, b70_wait_ns{0}, b70_rtt_ns{0}, b70_err{0}, b70_rtt_max_ns{0};
    std::atomic<long long> reply_ns{0}, plan_ns{0}, ctl_busy_ns{0}, cpu_busy_ns{0}, cpu_wait_land_ns{0}, read_ms_x1000{0},
        stage_wait_ns{0}, stage_job_ns{0}, wb_bulk{0};
    std::atomic<long long> pf_keys{0}, pf_skip{0}, pf_used{0}, pf_wasted{0};   // prefetch precision counters
    std::atomic<long long> promoted{0};   // in-flight prefetches promoted to demand priority
    std::atomic<long long> wb_early{0};   // write-backs drained by the wb thread at nv_copy completion
    std::atomic<long long> partial_lands{0}, partial_binds{0};
    std::atomic<long long> wb_lat_us{0}, wb_lat_max_us{0}, ring_low{0}, ring_low_vfree{0};
    std::atomic<long long> ring_borrowed{0};   // RAM slots lent to the prefill ring (cumulative)
    std::atomic<long long> stg_issued{0}, stg_ready{0}, stg_hit{0}, stg_late{0}, stg_nofree{0}, stg_dup{0}, stg_waste{0};
    std::atomic<long long> lane_skip{0}, lossy_calls{0}, lossy_tok{0}, lossy_tok_skip{0}, lossy_cold_x1e6{0},
        lossy_skip_x1e6{0}, lossy_coldu{0}, lossy_want{0}, lossy_bg_reads{0};   // BCP
};

struct Engine;
static Engine* G = nullptr;

struct StageEntry { int li; int j; int key; int ring; std::atomic<int> ready{0}; };

struct Engine
{
    // geometry
    int L = 0, E = 0, K = 0, H = 4096, I = 2048;
    int64_t rec_bytes = 0, slot_bytes = 0, piece = 0, dpiece = 0, off_u = 0, off_d = 0;   // dpiece: decode reads (GLM53_NV_DPIECE_KB)
    int fd = -1;
    int64_t ph = 0;
    // mapped region
    volatile int64_t* hc = nullptr; Req* req = nullptr; Rep* rep = nullptr; LogE* wbl = nullptr; LogE* adml = nullptr;
    int32_t* land = nullptr; volatile int64_t* home = nullptr; volatile int32_t* ram_res = nullptr; uint16_t* hx = nullptr; float* hout = nullptr;
    // RAM tier
    int nram = 0; std::vector<int64_t> ram_addr;
    std::vector<int32_t> slot_of; std::vector<int8_t> kst; std::vector<double> karr;
    std::vector<int32_t> owner, pinc, lprev, lnext; int lru_old = -1, lru_new = -1;
    std::vector<int32_t> freel;
    std::vector<uint8_t> prot; std::vector<int32_t> protl;
    // BCP lossy verify (GLM53_NV_LOSSY=1 or ext().set_lossy; approximate, not lossless): see lossy_plan
    bool lossy_on = false; float lossy_eps = 0.0f, lossy_whi = 2.0f; int lossy_renorm = 0, lossy_pf = 0, lossy_bg = 0;
    int lossy_lo = 0, lossy_hi = 1 << 30;
    int tail_r = 0, tail_scope = 0;   // Strata tail skip (patch_tail.py); tail_r 0 = off
    std::vector<uint8_t> lwant; std::vector<int16_t> luix;
    std::vector<uint8_t> pfs; int pf_max = 4;   // prefetched and not yet demanded; GLM53_NV_PF_MAX (4 = best measured)
    int pf_multi = 2;                           // GLM53_NV_PF_MULTI: prefetch hints per token of a verify window
    std::vector<float> score;
    std::mutex mu; std::condition_variable land_cv;
    int64_t land_tail = 0;
    struct WbP { int key, s, v; bool issued = false; double t = 0; };
    std::deque<WbP> wb_q; std::mutex wb_mu; std::condition_variable wb_cv; std::atomic<int> wb_inflight{0};
    std::thread wb_th;
    std::vector<int32_t> kvr;   // key -> VRAM ring slot backing it (write-back D2H pending) or -1
    std::vector<int> vfree; std::vector<int64_t> vring_addr; std::vector<cudaEvent_t> vr_ev;
    int land_target = 64, free_lo = 96;
    bool wb_early = true;   // GLM53_NV_WB_EARLY: drain parked victims at nv_copy completion (HC_CPDONE), restock at once
    bool rzc1 = true;       // GLM53_NV_RZC1: verify windows zero-copy (do not admit) RAM picks used by one token only
    bool exclusive = true;
    // NVMe channel model (plan arrival estimates)
    double nv_free_at = 0;
    // threads
    Reader rd;
    std::thread ctl_th, cpu_th;
    std::atomic<bool> quit{false};
    int64_t next_seq = 1;
    long long t_reply = 0;
    std::vector<long long> dbg_tn = std::vector<long long>(DBG, 0), dbg_tr = std::vector<long long>(DBG, 0), dbg_seq = std::vector<long long>(DBG, 0);
    // per-request host timeline (index seq % DBG): CPU job start / end, demand reads, prefetch reads, in-flight hits,
    // last landing of this request's demand reads (all CLOCK_REALTIME ns)
    std::vector<long long> dbg_cs = std::vector<long long>(DBG, 0), dbg_ce = std::vector<long long>(DBG, 0),
        dbg_nr = std::vector<long long>(DBG, 0), dbg_npf = std::vector<long long>(DBG, 0),
        dbg_inf = std::vector<long long>(DBG, 0), dbg_lm = std::vector<long long>(DBG, 0);
    std::vector<int64_t> kseq;      // key -> seq of the request that issued its read (> 0 demand, < 0 prefetch)
    // partial landing (GLM53_NV_PARTIAL): gu_res[key] = 1 once the gate+up bytes of an in-flight read have landed
    bool partial = true; std::vector<uint8_t> gu_res;
    static void on_partial(void* ctx, int64_t tag)
    {
        Engine* e = static_cast<Engine*>(ctx);
        { std::lock_guard<std::mutex> lk(e->mu); e->gu_res[int(tag)] = 1; e->C.partial_lands++; }
        e->land_cv.notify_all();
    }
    // layer-ahead VRAM staging (GLM53_NV_STG pool slots): at plan(li) the predicted RAM-resident experts of li+1 are DMA'd
    // into a host-managed VRAM pool once layer li's own copy is done (HC_CPDONE); at plan(li+1) a staged pick takes lane
    // LN_STG and nv_copy admits it with a D2D copy from the pool instead of a PCIe gather. Expert bytes are immutable, so a
    // staged copy stays valid while its RAM copy is resident; it is dropped on admission or RAM eviction.
    int stg_n = 0, stg_max = 3;
    // GLM53_NV_STG_GATE: when the staging DMA for layer li + 1 may start. 0: after layer li's copy kernel (HC_CPDONE);
    // 1: after its MoE kernel (HC_MOEDONE); 2 (default): after the copy kernel unless layer li's MoE reads an expert
    // zero-copy over PCIe (NZC / RZC lanes), which the DMA would slow down, then after the MoE kernel
    int stg_gate = 2;
    std::vector<int64_t> stg_addr, stg_use, stg_after; std::vector<int> stg_key, stg_rslot; std::vector<int8_t> stg_st, stg_gt;
    std::vector<long long> stg_t; std::vector<int32_t> staged_of; std::vector<cudaEvent_t> stg_ev;
    cudaStream_t stg_stream = nullptr; std::thread stg_th; std::mutex stg_mu; std::condition_variable stg_cv;
    std::deque<int> stg_q; long long stg_clock = 0;
    enum { SG_FREE = 0, SG_QUEUED = 1, SG_FLIGHT = 2, SG_READY = 3 };
    void stg_release(int j)   // caller holds mu; the slot must not be in flight
    {
        if (stg_key[j] >= 0 && staged_of[stg_key[j]] == j) staged_of[stg_key[j]] = -1;
        stg_key[j] = -1; stg_st[j] = SG_FREE;
    }
    void stg_drop_key(int key)   // caller holds mu: the staged copy of key is no longer wanted
    {
        const int j = staged_of[key];
        if (j < 0) return;
        staged_of[key] = -1;
        if (stg_st[j] == SG_READY || stg_st[j] == SG_FREE) { stg_key[j] = -1; stg_st[j] = SG_FREE; }
        else stg_key[j] = -1;   // queued / in flight: freed by the staging thread when its DMA completes
    }
    int stg_alloc(int64_t s)   // caller holds mu: a free pool slot, else the least recently used ready one not in use
    {
        int best = -1;
        for (int j = 0; j < stg_n; ++j)
        {
            if (stg_st[j] == SG_FREE) return j;
            if (stg_st[j] == SG_READY && stg_use[j] < s && !(stg_key[j] >= 0 && prot[stg_key[j]]) && (best < 0 || stg_t[j] < stg_t[best])) best = j;
        }
        if (best >= 0) { if (stg_use[best] < 0) C.stg_waste++; stg_release(best); }
        return best;
    }
    void stg_loop()
    {
        pin_self(wb_cpu);
        cudaSetDevice(dev_index);
        std::vector<int> infl, go;
        while (!quit.load())
        {
            go.clear();
            {
                std::unique_lock<std::mutex> lq(stg_mu);
                if (stg_q.empty() && infl.empty()) stg_cv.wait_for(lq, std::chrono::milliseconds(20));
                const int64_t cpd = __atomic_load_n(&hc[HC_CPDONE], __ATOMIC_ACQUIRE);
                const int64_t mod = __atomic_load_n(&hc[HC_MOEDONE], __ATOMIC_ACQUIRE);
                while (!stg_q.empty() && stg_after[stg_q.front()] <= (stg_gt[stg_q.front()] ? mod : cpd))
                { go.push_back(stg_q.front()); stg_q.pop_front(); }
            }
            for (int j : go)
            {
                int rs;
                { std::lock_guard<std::mutex> lk(mu); rs = stg_rslot[j]; stg_st[j] = SG_FLIGHT; }
                cudaMemcpyAsync((void*) stg_addr[j], (const void*) ram_addr[rs], size_t(slot_bytes), cudaMemcpyHostToDevice, stg_stream);
                cudaEventRecord(stg_ev[j], stg_stream);
                infl.push_back(j);
            }
            for (size_t i = 0; i < infl.size();)
            {
                const int j = infl[i];
                if (cudaEventQuery(stg_ev[j]) != cudaSuccess) { ++i; continue; }
                {
                    std::lock_guard<std::mutex> lk(mu);
                    if (stg_rslot[j] >= 0) { pinc[stg_rslot[j]]--; stg_rslot[j] = -1; }
                    stg_st[j] = stg_key[j] >= 0 ? SG_READY : SG_FREE;
                    C.stg_ready++;
                }
                infl[i] = infl.back(); infl.pop_back();
            }
            bool busy;
            { std::lock_guard<std::mutex> lq(stg_mu); busy = !stg_q.empty(); }
            if (busy || !infl.empty()) std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }
    int64_t cur_seq = 0; int cur_nr = 0, cur_npf = 0, cur_inf = 0;
    // routing trace (decode-class requests): [0x7EAC, li, ntok, np, nu, pe[np], pw bits[np], key * 4 + state[nu]]
    bool trace_on = false; std::vector<int32_t> trace; std::mutex trace_mu;
    int64_t wb_read = 0, adm_read = 0;
    int ctl_cpu = -1, dev_index = 0, wb_cpu = -1;
    Pol pol;
    Counters C;
    // CPU tier
    bool cpu_ready = false;
    Pool pool;
    std::vector<Layer> layers;
    std::vector<torch::Tensor> keep;
    int cpu_mode = -1;
    std::atomic<int64_t> cpu_job_seq{0};
    struct CJob { int64_t seq; int li, ntok, np; int e[MAXP], t[MAXP], lane[MAXP]; float w[MAXP]; } cj;
    // N137 B70 expert tier: static resident key set on the B70 server, shared host ring (b70/ring.py layout)
    std::vector<int8_t> b70k; bool b70_on = false; uint8_t* b70r = nullptr; int64_t b70_seq = 0; int64_t b70_timeout_ns = 2000000000LL;
    std::vector<float> xf, tmpout;
    // stage engine
    std::vector<int64_t> pf_ring_addr;
    std::vector<cudaEvent_t> pf_ev; std::vector<int> pf_ev_pending;
    // prefill ring borrowed from the RAM tier (GLM53_NV_PF_RING_BORROW, set_ring_borrow): stage_begin takes ring_n RAM
    // slots (free list first, else the LRU-oldest unpinned experts), stage_end returns them to the free list, so the
    // ring's memory serves the RAM tier during decode
    int ring_n = 0; std::vector<int> st_ring_slots; std::vector<int64_t> st_ring_addr;
    const std::vector<int64_t>& ring_addr() const { return ring_n > 0 ? st_ring_addr : pf_ring_addr; }
    std::vector<std::vector<int64_t>> st_ram_src, st_ram_j;    // per layer: src addr, staging index
    std::vector<std::vector<int>> st_nv_idx;                    // per layer: indices into st_nv
    std::deque<StageEntry> st_nv;
    std::vector<int32_t> st_pins;
    std::thread st_th; std::atomic<bool> st_cancel{false}; std::atomic<int> st_inflight{0};
    std::mutex st_mu; std::condition_variable st_cv;
    int st_qd = 24;
    bool st_active = false;
    cudaStream_t wb_stream = nullptr;

    // ---------------- LRU over slots ----------------
    void lru_remove(int s)
    {
        const int p = lprev[s], n = lnext[s];
        if (p >= 0) lnext[p] = n; else if (lru_old == s) lru_old = n;
        if (n >= 0) lprev[n] = p; else if (lru_new == s) lru_new = p;
        lprev[s] = lnext[s] = -1;
    }
    void lru_push_new(int s) { lprev[s] = lru_new; lnext[s] = -1; if (lru_new >= 0) lnext[lru_new] = s; lru_new = s; if (lru_old < 0) lru_old = s; }
    void lru_push_old(int s) { lnext[s] = lru_old; lprev[s] = -1; if (lru_old >= 0) lprev[lru_old] = s; lru_old = s; if (lru_new < 0) lru_new = s; }
    void lru_touch(int s) { if (lru_new == s) return; lru_remove(s); lru_push_new(s); }

    void set_home(int key, int64_t a)
    {
        home[int64_t(key) * 3 + 0] = a; home[int64_t(key) * 3 + 1] = a + off_u; home[int64_t(key) * 3 + 2] = a + off_d;
    }
    // resident -> none (caller holds mu). Order: device-visible flag first, then home, then host state.
    int evict_key(int key)
    {
        const int s = slot_of[key];
        if (pfs[key]) { C.pf_wasted++; pfs[key] = 0; }   // prefetched, evicted before any demand use
        if (stg_n) stg_drop_key(key);
        __atomic_store_n(&ram_res[key], 0, __ATOMIC_RELEASE);
        set_home(key, ph);
        kst[key] = KS_NONE; slot_of[key] = -1; owner[s] = SL_FREE;
        lru_remove(s);
        C.evictions++;
        return s;
    }
    // one free slot (caller holds mu): free list, else the oldest evictable resident key; -1 if none
    int take_slot()
    {
        if (!freel.empty()) { int s = freel.back(); freel.pop_back(); return s; }
        int guard = 0;
        for (int s = lru_old; s >= 0 && guard < nram; s = lnext[s], ++guard)
        {
            const int key = owner[s];
            if (key < 0 || kst[key] != KS_RES || prot[key] || pinc[s] > 0) continue;
            return evict_key(key);
        }
        C.alloc_fail++;
        return -1;
    }
    void free_slot(int s) { owner[s] = SL_FREE; freel.push_back(s); }

    // landed (reader thread)
    static void on_land(void* ctx, int64_t tag, int err, double ms)
    {
        Engine* e = static_cast<Engine*>(ctx);
        const int key = int(tag);
        e->C.read_ms_x1000 += (long long) (ms * 1000);
        if (err) { e->C.read_err++; e->hc[HC_ERR] = 16; fprintf(stderr, "nv2: read error %d key %d\n", err, key); }
        {
            std::lock_guard<std::mutex> lk(e->mu);
            {
                const int64_t sq = e->kseq[key];
                if (sq > 0) { long long& lm = e->dbg_lm[sq % DBG]; lm = std::max(lm, realtime_ns()); }
                e->kseq[key] = 0;
            }
            if (e->kst[key] == KS_INFLIGHT)
            {
                e->kst[key] = KS_RES;
                const int s = e->slot_of[key];
                e->lru_push_new(s);
                std::atomic_thread_fence(std::memory_order_seq_cst);
                __atomic_store_n(&e->ram_res[key], 1, __ATOMIC_RELEASE);
            }
        }
        e->land_cv.notify_all();
    }
    // start an NVMe read of key into a fresh slot (caller holds mu). Returns false if no slot. cls: RC_DEMAND /
    // RC_PREFETCH / RC_BULK (reader priority classes); only demand reads advance the planner's arrival model
    bool start_read(int key, int cls, double* arrival)
    {
        const int s = take_slot();
        if (s < 0) return false;
        owner[s] = key; slot_of[key] = s; kst[key] = KS_INFLIGHT;
        set_home(key, ram_addr[s]);                 // device reads home only after ram_res == 1
        const double t = now_ms();
        if (cls == RC_DEMAND || !rd.prio)
        {
            const double start = std::max(t, nv_free_at);
            double svc = pol.nvdeep;
            if (start > nv_free_at || nv_free_at == 0) svc += pol.nvlat;
            nv_free_at = start + svc;
            karr[key] = nv_free_at;
        }
        else karr[key] = std::max(t, nv_free_at) + pol.nvdeep + pol.nvlat;
        if (arrival) *arrival = karr[key];
        if (partial) gu_res[key] = 0;
        rd.submit(fd, int64_t(key) * rec_bytes, ram_addr[s], slot_bytes, cls == RC_BULK ? piece : dpiece, cls, &Engine::on_land, this, key, key,
                  (partial && cls != RC_BULK) ? &Engine::on_partial : nullptr, off_d);
        C.reads++; C.read_bytes += slot_bytes;
        return true;
    }

    // ---------------- logs (exclusive tier) ----------------
    void take_wb_entry(const LogE& le)   // caller holds mu
    {
        const int key = le.key, v = le.slot;
        if (kst[key] != KS_NONE) { vfree.push_back(v); C.wb_dup++; return; }
        const int s = take_slot();
        if (s < 0) { vfree.push_back(v); C.wb_drop++; return; }
        // resident at once, backed by the VRAM ring slot (valid bytes, device-readable); the wb thread copies it
        // to RAM slot s on a copy engine and then switches home to s. Pinned until then (no eviction).
        owner[s] = key; slot_of[key] = s; kst[key] = KS_RES; kvr[key] = v; pinc[s]++;
        lru_push_new(s);
        set_home(key, vring_addr[v]);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        __atomic_store_n(&ram_res[key], 1, __ATOMIC_RELEASE);
        {
            std::lock_guard<std::mutex> lw(wb_mu);
            wb_q.push_back({ key, s, v, false, now_ms() });
        }
        wb_cv.notify_one();
        C.wb_issued++;
    }
    // early write-back drain (wb thread, caller holds mu): victims parked by nv_copy calls that have finished
    // (HC_CPDONE) start their D2H now instead of at the next request's serve. Only write-back entries that precede every
    // pending admission entry in log order are taken, so the (wb before adm of the same seq) order of drain_logs holds.
    int drain_wb_early()
    {
        if (!exclusive) return 0;
        const int64_t cp = __atomic_load_n(&hc[HC_CPDONE], __ATOMIC_ACQUIRE);
        const int64_t wbn = __atomic_load_n(&hc[HC_WBN], __ATOMIC_ACQUIRE);
        const int64_t admn = __atomic_load_n(&hc[HC_ADMN], __ATOMIC_ACQUIRE);
        int n = 0;
        while (wb_read < wbn && wbl[wb_read % LG].seq <= cp)
        {
            if (adm_read < admn && adml[adm_read % LG].seq < wbl[wb_read % LG].seq) break;
            const LogE le = wbl[wb_read % LG]; wb_read++;
            take_wb_entry(le);
            ++n;
        }
        C.wb_early += n;
        return n;
    }
    void drain_logs(int64_t upto)   // entries with seq < upto are complete (their kernels ran before request upto)
    {
        const int64_t wbn = __atomic_load_n(&hc[HC_WBN], __ATOMIC_ACQUIRE);
        const int64_t admn = __atomic_load_n(&hc[HC_ADMN], __ATOMIC_ACQUIRE);
        while (true)
        {
            const bool hw = wb_read < wbn && wbl[wb_read % LG].seq < upto;
            const bool ha = adm_read < admn && adml[adm_read % LG].seq < upto;
            if (!hw && !ha) break;
            bool take_wb = hw && (!ha || wbl[wb_read % LG].seq <= adml[adm_read % LG].seq);
            if (take_wb)
            {
                // VRAM victim parked in VRAM ring slot le.slot by nv_copy (complete: its kernel ran before request upto)
                const LogE le = wbl[wb_read % LG]; wb_read++;
                take_wb_entry(le);
            }
            else
            {
                const LogE le = adml[adm_read % LG]; adm_read++;
                const int key = le.key;
                if (stg_n) stg_drop_key(key);
                if (!exclusive || kst[key] != KS_RES) continue;
                const int s = slot_of[key];
                if (pinc[s] > 0 || prot[key]) { C.adm_pinned++; lru_remove(s); lru_push_old(s); continue; }
                if (pfs[key]) { C.pf_used++; pfs[key] = 0; }   // admitted to VRAM: the prefetch was used
                free_slot(evict_key(key));
                C.evictions--;   // not a capacity eviction
                C.adm_freed++;
            }
        }
    }
    void stock_landing()
    {
        const int64_t used = __atomic_load_n(&hc[HC_WBN], __ATOMIC_ACQUIRE);   // ring slots popped by the device
        while (land_tail - used < land_target && land_tail - used < LR && !vfree.empty())
        {
            land[land_tail % LR] = vfree.back(); vfree.pop_back();
            land_tail++;
            C.land_pushed++;
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        __atomic_store_n(&hc[HC_LANDT], land_tail, __ATOMIC_RELEASE);
    }
    // write-back thread: the only thread besides Python that makes CUDA calls on the decode path. Nothing on the
    // device waits for it (vring-backed keys are device-readable), so a driver lock held elsewhere cannot deadlock.
    void wb_loop()
    {
        pin_self(wb_cpu);
        cudaSetDevice(dev_index);
        std::deque<WbP> pend;
        while (!quit.load())
        {
            if (wb_early)
            {
                const bool logs = wb_read < __atomic_load_n(&hc[HC_WBN], __ATOMIC_ACQUIRE);
                if (logs) { std::lock_guard<std::mutex> lk(mu); drain_wb_early(); }
            }
            {
                std::unique_lock<std::mutex> lw(wb_mu);
                if (wb_q.empty() && pend.empty())
                {
                    // early drain on: nap briefly (a parked victim is noticed within ~0.1 ms of its nv_copy); else wait
                    if (wb_early) wb_cv.wait_for(lw, std::chrono::microseconds(100));
                    else wb_cv.wait_for(lw, std::chrono::milliseconds(50));
                }
                while (!wb_q.empty()) { pend.push_back(wb_q.front()); wb_q.pop_front(); }
                wb_inflight.store((int) pend.size());
            }
            for (auto& w : pend)
                if (!w.issued)
                {
                    cudaMemcpyAsync((void*) ram_addr[w.s], (const void*) vring_addr[w.v], size_t(slot_bytes), cudaMemcpyDeviceToHost, wb_stream);
                    cudaEventRecord(vr_ev[w.v], wb_stream);
                    w.issued = true;
                }
            int done = 0;
            while (!pend.empty() && cudaEventQuery(vr_ev[pend.front().v]) == cudaSuccess)
            {
                const WbP w = pend.front(); pend.pop_front(); done++;
                {
                    const long long us = (long long) ((now_ms() - w.t) * 1000);
                    C.wb_lat_us += us; C.wb_lat_max_us = std::max<long long>(C.wb_lat_max_us.load(), us);
                }
                std::lock_guard<std::mutex> lk(mu);
                if (kst[w.key] == KS_RES && slot_of[w.key] == w.s && kvr[w.key] == w.v)
                {
                    set_home(w.key, ram_addr[w.s]);
                    kvr[w.key] = -1;
                    C.wb_landed++;
                }
                pinc[w.s]--;
                vfree.push_back(w.v);
                if (wb_early) stock_landing();   // recycled landing slots reach the device before the next serve
            }
            {
                std::lock_guard<std::mutex> lw(wb_mu);
                wb_inflight.store((int) pend.size());
            }
            if (!pend.empty() && !done) std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }
    void stock_free()
    {
        int guard = 0;
        while ((int) freel.size() < free_lo && guard++ < 16)
        {
            int s = -1;
            for (int x = lru_old, g = 0; x >= 0 && g < nram; x = lnext[x], ++g)
            {
                const int key = owner[x];
                if (key < 0 || kst[key] != KS_RES || prot[key] || pinc[x] > 0) continue;
                s = evict_key(key); break;
            }
            if (s < 0) break;
            freel.push_back(s);
        }
    }

    // ---------------- plan_layer (moetier semantics) ----------------
    struct PK { int u, key, cnt; double arr; float sc; };
    std::vector<PK> V, Rm, N;
    bool lossy_cold(const Req& r, int u) const
    {
        const int key = r.key[u];
        return r.cls[u] != LN_VRAM && kst[key] == KS_NONE && !(b70_on && b70k[key]);
    }
    // BCP: per token, cold picks sorted by weight; fetch largest-first while the token's remaining cold mass > eps
    // (fraction of its routed mass) or the pick's own fraction >= whi. The union over tokens is what gets read, so a
    // cold expert fetched for one token is exact for every token that picked it.
    void lossy_plan(const Req& r, Rep& p)
    {
        const int topk = std::max(1, r.np / std::max(1, r.ntok));
        p.rn_on = lossy_renorm; p.rn_topk = topk;
        for (int t = 0; t < MAXB; ++t) p.rn_sc[t] = 1.0f;
        long long ncu = 0;
        for (int u = 0; u < r.nu; ++u)
        {
            lwant[u] = 0;
            const int e = r.key[u] - r.li * E;
            if (e >= 0 && e < E) luix[e] = (int16_t) u;
            ncu += lossy_cold(r, u);
        }
        for (int t = 0; t < r.ntok; ++t)
        {
            float cw[MAXP]; int cu[MAXP]; int nc = 0; float W = 0.f;
            for (int j = 0; j < topk; ++j)
            {
                const int i = t * topk + j;
                if (i >= r.np) break;
                const float w = r.pw[i]; W += w;
                const int e = r.pe[i];
                if (e < 0 || e >= E) continue;
                const int u = luix[e];
                if (u < 0 || !lossy_cold(r, u)) continue;
                int k = nc++;
                while (k > 0 && cw[k - 1] < w) { cw[k] = cw[k - 1]; cu[k] = cu[k - 1]; --k; }
                cw[k] = w; cu[k] = u;
            }
            if (W <= 0.f || nc == 0) continue;
            float m = 0.f;
            for (int k = 0; k < nc; ++k) m += cw[k];
            C.lossy_cold_x1e6 += (long long) (1e6f * m / W);
            for (int k = 0; k < nc; ++k)
                if (m > lossy_eps * W || cw[k] >= lossy_whi * W) { lwant[cu[k]] = 1; m -= cw[k]; }
        }
        for (int t = 0; t < r.ntok; ++t)   // after the union: skipped mass per token, renorm scales
        {
            float W = 0.f, S = 0.f;
            for (int j = 0; j < topk; ++j)
            {
                const int i = t * topk + j;
                if (i >= r.np) break;
                const float w = r.pw[i]; W += w;
                const int e = r.pe[i];
                if (e < 0 || e >= E) continue;
                const int u = luix[e];
                if (u >= 0 && lossy_cold(r, u) && !lwant[u]) S += w;
            }
            C.lossy_tok++;
            C.lossy_skip_x1e6 += (long long) (W > 0.f ? 1e6f * S / W : 0.f);
            if (S > 0.f) C.lossy_tok_skip++;
            if (t < MAXB) p.rn_sc[t] = (W - S > 1e-6f) ? W / (W - S) : 1.0f;
        }
        long long nw = 0;
        for (int u = 0; u < r.nu; ++u) nw += lwant[u];
        C.lossy_coldu += ncu; C.lossy_want += nw; C.lossy_calls++;
    }
    // Strata tail skip (set_tail / GLM53_NV_TAIL): a pick is skipped only when every token of the call routes it at
    // rank >= tail_r of its top-k (rank by routing weight, 0 = best); scope 1 also covers RAM-tier picks. No renormalisation.
    void lossy_tail_plan(const Req& r, Rep& p)
    {
        const int topk = std::max(1, r.np / std::max(1, r.ntok));
        p.rn_on = 0; p.rn_topk = topk;
        for (int t = 0; t < MAXB; ++t) p.rn_sc[t] = 1.0f;
        for (int u = 0; u < r.nu; ++u)
        {
            const int e = r.key[u] - r.li * E;
            if (e >= 0 && e < E) luix[e] = (int16_t) u;
        }
        int minr[MAXU];
        for (int u = 0; u < r.nu; ++u) minr[u] = 1 << 20;
        for (int t = 0; t < r.ntok; ++t)
            for (int j = 0; j < topk; ++j)
            {
                const int i = t * topk + j;
                if (i >= r.np) break;
                const int e = r.pe[i];
                if (e < 0 || e >= E) continue;
                const int u = luix[e];
                if (u < 0) continue;
                const float w = r.pw[i];
                int rank = 0;
                for (int j2 = 0; j2 < topk; ++j2)
                {
                    if (j2 == j || t * topk + j2 >= r.np) continue;
                    const float w2 = r.pw[t * topk + j2];
                    if (w2 > w || (w2 == w && j2 < j)) ++rank;
                }
                minr[u] = std::min(minr[u], rank);
            }
        long long nw = 0;
        for (int u = 0; u < r.nu; ++u)
        {
            const int key = r.key[u];
            const bool cold = lossy_cold(r, u);
            const bool ram = tail_scope == 1 && r.cls[u] != LN_VRAM && kst[key] == KS_RES && kvr[key] < 0;
            lwant[u] = ((cold || ram) && minr[u] >= tail_r) ? 0 : 1;
            nw += lwant[u];
        }
        C.lossy_calls++; C.lossy_want += nw;
    }
    void lossy_bg_reads(const Req& r)   // heaviest skipped experts first, low priority (behind demand reads)
    {
        float bw[MAXU]; int bk[MAXU]; int nb = 0;
        for (int u = 0; u < r.nu && nb < MAXU; ++u)
        {
            if (!lossy_cold(r, u) || lwant[u]) continue;
            float w = 0.f;
            for (int i = 0; i < r.np; ++i) if (r.pe[i] == r.key[u] - r.li * E) w += r.pw[i];
            int k = nb++;
            while (k > 0 && bw[k - 1] < w) { bw[k] = bw[k - 1]; bk[k] = bk[k - 1]; --k; }
            bw[k] = w; bk[k] = r.key[u];
        }
        for (int k = 0; k < std::min(nb, lossy_bg); ++k)
            if (kst[bk[k]] == KS_NONE && start_read(bk[k], RC_PREFETCH, nullptr)) { C.lossy_bg_reads++; pfs[bk[k]] = 1; }
    }
    void plan_and_reply(const Req& r, Rep& p)
    {
        const double t0 = now_ms();
        const bool dec = r.kind == RK_DECODE || r.kind == RK_VERIFY;
        const bool lossy = lossy_on && r.kind == RK_VERIFY && r.np <= MAXP && r.li >= lossy_lo && r.li <= lossy_hi;
        p.rn_on = 0; p.rn_topk = 8;
        cur_nr = cur_npf = cur_inf = 0;
        if (trace_on && dec && r.np <= MAXP)
        {
            std::lock_guard<std::mutex> lt(trace_mu);
            if (trace.size() < (size_t(1) << 26))
            {
                trace.push_back(0x7EAC); trace.push_back(r.li); trace.push_back(r.ntok); trace.push_back(r.np); trace.push_back(r.nu);
                for (int i = 0; i < r.np; ++i) trace.push_back(r.pe[i]);
                for (int i = 0; i < r.np; ++i) { int32_t b; std::memcpy(&b, &r.pw[i], 4); trace.push_back(b); }
                for (int u = 0; u < r.nu; ++u)
                {
                    const int key = r.key[u];
                    const int stt = r.cls[u] == LN_VRAM ? 3 : int(kst[key]);
                    trace.push_back(key * 4 + stt);
                }
                if (r.has_lg)
                {
                    // 0x7EAD: + logits; 0x7EAE: + logits + layer-ahead predicted logits for li + 1
                    trace[trace.size() - (5 + 2 * r.np + r.nu)] = r.has_plg ? 0x7EAE : 0x7EAD;
                    for (int e2 = 0; e2 < E; e2 += 2) trace.push_back(int32_t(uint32_t(r.lg[e2]) | (uint32_t(r.lg[e2 + 1]) << 16)));
                    if (r.has_plg)
                        for (int e2 = 0; e2 < E; e2 += 2) trace.push_back(int32_t(uint32_t(r.plg[e2]) | (uint32_t(r.plg[e2 + 1]) << 16)));
                }
            }
        }
        if (lossy && tail_r > 0) lossy_tail_plan(r, p);
        else if (lossy) lossy_plan(r, p);
        const bool cpu = dec && pol.cpu_on && cpu_ready && r.np <= MAXP && r.ntok <= MAXB;
        V.clear(); Rm.clear(); N.clear();
        int nnv = 0, nb70 = 0, nstg = 0;
        const bool b70 = dec && b70_on && cpu_ready && r.np <= MAXP && r.ntok <= MAXB;   // independent of the CPU lane switch
        for (int u = 0; u < r.nu; ++u)
        {
            const int key = r.key[u];
            if (r.cls[u] == LN_VRAM) { p.lane[u] = LN_VRAM; V.push_back({ u, key, r.cnt[u], 0, 0 }); continue; }
            if (b70 && b70k[key]) { p.lane[u] = LN_B70; nb70++; C.lane_b70++; continue; }
            if (pfs[key]) { C.pf_used++; pfs[key] = 0; }   // a prefetched expert is demanded (resident or in flight)
            if (kst[key] == KS_RES)
            {
                lru_touch(slot_of[key]);
                if (stg_n && dec && staged_of[key] >= 0 && kvr[key] < 0)
                {
                    const int j = staged_of[key];
                    if (stg_st[j] == SG_READY) { p.lane[u] = LN_STG; p.stg[u] = j; stg_use[j] = cur_seq; stg_t[j] = ++stg_clock; C.stg_hit++; nstg++; C.ram++; continue; }
                    C.stg_late++;
                }
                if (kvr[key] >= 0) { p.lane[u] = LN_RAM; C.lane_zc++; C.vring_hits++; if (dec) C.ram++; continue; }   // VRAM-ring backed: GPU lane
                if (lossy && tail_r > 0 && tail_scope == 1 && !lwant[u]) { p.lane[u] = LN_SKIP; C.lane_skip++; continue; }
                Rm.push_back({ u, key, r.cnt[u], 0, score[key] });
            }
            else
            {
                if (lossy && kst[key] == KS_NONE && !lwant[u]) { p.lane[u] = LN_SKIP; C.lane_skip++; continue; }
                double a = 0;
                if (kst[key] == KS_INFLIGHT) { a = karr[key]; C.inflight_hit++; cur_inf++; if (rd.promote(key)) C.promoted++; }
                else if (!start_read(key, RC_DEMAND, &a)) { a = 1e9; hc[HC_ERR] = 19; fprintf(stderr, "nv2: no RAM slot for key %d\n", key); }
                else { kseq[key] = cur_seq; cur_nr++; }
                N.push_back({ u, key, r.cnt[u], a - t0, score[key] });
                nnv++;
            }
        }
        if (dec) { C.picks += r.nu; C.vram += V.size(); C.ram += Rm.size(); C.nvme += N.size(); }
        double g = (V.empty() && !nstg) ? 0.0 : pol.g0 + pol.thit * (V.size() + nstg);
        double c = (cpu && (!Rm.empty() || !N.empty())) ? pol.ca : 0.0;
        int ncpu = 0;
        std::sort(Rm.begin(), Rm.end(), [](const PK& a, const PK& b) { return a.cnt != b.cnt ? a.cnt > b.cnt : (a.sc != b.sc ? a.sc < b.sc : a.key < b.key); });
        std::sort(N.begin(), N.end(), [](const PK& a, const PK& b) { return a.arr < b.arr; });
        for (auto& k : Rm)
        {
            if (!dec) { p.lane[k.u] = LN_RAM; C.lane_zc++; continue; }
            const double ce = c + pol.cb + pol.ctok * (k.cnt - 1);
            const double ge = g + pol.tzc;
            if (cpu && ncpu < pol.maxcpu && std::max(ce, g) <= std::max(c, ge)) { c = ce; p.lane[k.u] = LN_CPU; ncpu++; C.lane_cpu++; }
            // verify window: an expert only one of its tokens picked is read once either way; zero-copy it instead of
            // admitting it, which keeps the VRAM cache and the write-back landing ring for the shared ones
            else if (rzc1 && r.ntok > 1 && k.cnt == 1) { g = ge; p.lane[k.u] = LN_RZC; C.lane_zc++; }
            else { g = ge; p.lane[k.u] = LN_RAM; C.lane_zc++; }
        }
        for (auto& k : N)
        {
            const double a = std::max(0.0, k.arr);
            const double ce = std::max(c, a) + pol.cb + pol.ctok * (k.cnt - 1);
            // cold: zero-copy, keep VRAM. A verify window admits NVMe picks shared by several of its tokens instead (the
            // fused decode kernel reads a zero-copy expert once per token that picked it)
            const bool nzc = dec && pol.nv_noadmit && (r.ntok == 1 || k.cnt == 1);
            // GPU cost after landing: zero-copy in the MoE kernel (tnzc, GLM53_NV_TNZC; < 0: legacy push + thit model)
            const double ge = (nzc && pol.tnzc >= 0) ? std::max(g, a) + pol.tnzc : std::max(g, a + pol.push) + pol.thit;
            if (cpu && pol.nvcpu_on && ncpu < pol.maxcpu && std::max(ce, g) <= std::max(c, ge)) { c = ce; p.lane[k.u] = LN_NVC; ncpu++; C.lane_nvc++; }
            else if (nzc) { g = pol.tnzc >= 0 ? ge : std::max(g, a) + pol.tzc; p.lane[k.u] = LN_NZC; C.lane_nvg++; }
            else { g = ge; p.lane[k.u] = LN_NVG; C.lane_nvg++; }
        }
        p.ncpu = ncpu + nb70; p.nnv = nnv;
        // layer-ahead prefetch hints (next layer's predicted picks): read the non-resident ones now (behind demand reads)
        // rank order: the first pf_max are the best; a verify window gets pf_multi per token (rows interleaved by rank)
        const int npf_lim = std::min(std::min(r.npf, (int) MAXP), r.ntok > 1 ? pf_multi * r.ntok : pf_max);
        if (lossy && lossy_bg > 0) lossy_bg_reads(r);
        for (int i = 0; i < (lossy ? std::min(npf_lim, lossy_pf) : npf_lim); ++i)
        {
            const int key = r.pf_key[i];
            C.pf_keys++;
            if (key < 0 || key >= K || kst[key] != KS_NONE || (b70_on && b70k[key])) { C.pf_skip++; continue; }
            if (start_read(key, RC_PREFETCH, nullptr)) { C.prefetch++; pfs[key] = 1; kseq[key] = -cur_seq; cur_npf++; }
        }
        if (stg_n && dec && r.ntok == 1)
        {
            bool zc_moe = false;
            for (int u = 0; u < r.nu; ++u) if (p.lane[u] == LN_NZC || p.lane[u] == LN_RZC) { zc_moe = true; break; }
            int ns = 0;
            for (int i = 0; i < std::min(r.npf, (int) MAXP) && ns < stg_max; ++i)
            {
                const int key = r.pf_key[i];
                if (key < 0 || key >= K || kst[key] != KS_RES || kvr[key] >= 0) continue;
                if (staged_of[key] >= 0) { C.stg_dup++; ns++; stg_t[staged_of[key]] = ++stg_clock; continue; }
                const int j = stg_alloc(cur_seq);
                if (j < 0) { C.stg_nofree++; break; }
                const int rs = slot_of[key];
                pinc[rs]++;
                stg_key[j] = key; staged_of[key] = j; stg_st[j] = SG_QUEUED; stg_after[j] = cur_seq; stg_rslot[j] = rs;
                stg_gt[j] = stg_gate == 2 ? (zc_moe ? 1 : 0) : int8_t(stg_gate);
                stg_use[j] = -1; stg_t[j] = ++stg_clock;
                { std::lock_guard<std::mutex> lq(stg_mu); stg_q.push_back(j); }
                ns++; C.stg_issued++;
            }
            stg_cv.notify_one();
        }
        if (lossy) for (int u = 0; u < r.nu; ++u) { const int e = r.key[u] - r.li * E; if (e >= 0 && e < E) luix[e] = -1; }
        C.plan_ns += (long long) ((now_ms() - t0) * 1e6);
    }

    void serve(int64_t s)
    {
        const double t0 = now_ms();
        const Req& r = req[s % RQ];
        Rep& p = rep[s % RQ];
        {
            std::lock_guard<std::mutex> lk(mu);
            for (int k : protl) prot[k] = 0;      // request s-1 is complete on the device (its kernels ran before pub(s))
            protl.clear();
            for (int u = 0; u < r.nu; ++u) { prot[r.key[u]] = 1; protl.push_back(r.key[u]); }
            drain_logs(s);                         // write-back landings may evict: never this request's keys
            cur_seq = s;
            dbg_lm[s % DBG] = 0; dbg_cs[s % DBG] = 0; dbg_ce[s % DBG] = 0;
            plan_and_reply(r, p);
            dbg_nr[s % DBG] = cur_nr; dbg_npf[s % DBG] = cur_npf; dbg_inf[s % DBG] = cur_inf;
            C.reqs++; (r.kind != RK_SMALL ? C.dec_reqs : C.small_reqs)++; if (r.kind != RK_SMALL) C.dec_tok += r.ntok;
        }
        if (p.ncpu > 0)
        {
            cj.seq = s; cj.li = r.li; cj.ntok = r.ntok; cj.np = 0;
            for (int i = 0; i < r.np; ++i)
            {
                const int e = r.pe[i];
                if (e < 0 || e >= E) continue;
                int u = -1;
                for (int q = 0; q < r.nu; ++q) if (r.key[q] == r.li * E + e) { u = q; break; }
                if (u < 0) continue;
                const int ln = p.lane[u];
                if (ln != LN_CPU && ln != LN_NVC && ln != LN_B70) continue;
                const int topk = r.np / std::max(1, r.ntok);
                const int tt = i / std::max(1, topk);
                cj.e[cj.np] = e; cj.t[cj.np] = tt; cj.w[cj.np] = r.pw[i] * ((p.rn_on && tt < MAXB) ? p.rn_sc[tt] : 1.0f);
                cj.lane[cj.np] = ln; cj.np++;
            }
            cpu_job_seq.store(s, std::memory_order_release);
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        __atomic_store_n(&p.seq, s, __ATOMIC_RELEASE);
        __atomic_store_n(&hc[HC_ACK], s, __ATOMIC_RELEASE);
        t_reply = realtime_ns();
        C.reply_ns += (long long) ((now_ms() - t0) * 1e6);
        {
            std::lock_guard<std::mutex> lk(mu);
            stock_landing();
            stock_free();
            const int64_t ring = land_tail - __atomic_load_n(&hc[HC_WBN], __ATOMIC_ACQUIRE);
            if (ring < 8) { C.ring_low++; C.ring_low_vfree += (long long) vfree.size(); }
        }
        const long long dt = (long long) ((now_ms() - t0) * 1e6);
        C.ctl_busy_ns += dt; C.serve_max_ns = std::max<long long>(C.serve_max_ns.load(), dt);
    }

    void ctl_loop()
    {
        pin_self(ctl_cpu);
        double last = now_ms();
        long n = 0;
        while (!quit.load(std::memory_order_relaxed))
        {
            const int64_t s = next_seq;
            const Req& r = req[s % RQ];
            if (__atomic_load_n(&r.seq, __ATOMIC_ACQUIRE) != s)
            {
                cpu_relax();
                if ((++n & 1023) == 0 && now_ms() - last > 200.0) std::this_thread::sleep_for(std::chrono::microseconds(20));
                continue;
            }
            dbg_tn[s % DBG] = realtime_ns(); dbg_seq[s % DBG] = s;
            serve(s);
            dbg_tr[s % DBG] = t_reply;
            last = now_ms();
            next_seq = s + 1;
        }
    }

    // ---------------- CPU worker ----------------
    void cpu_loop(int cpu0)
    {
        Pool::pin(cpu0);
        int64_t last = cpu_job_seq.load();
        long idle = 0;
        while (!quit.load(std::memory_order_relaxed))
        {
            const int64_t s = cpu_job_seq.load(std::memory_order_acquire);
            if (s == last) { if (++idle < 400000) _mm_pause(); else std::this_thread::sleep_for(std::chrono::microseconds(20)); continue; }
            idle = 0;
            const double t0 = now_ms();
            dbg_cs[s % DBG] = realtime_ns();
            run_cpu_job();
            dbg_ce[s % DBG] = realtime_ns();
            C.cpu_busy_ns += (long long) ((now_ms() - t0) * 1e6);
            last = s;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            __atomic_store_n(&hc[HC_CDONE], s, __ATOMIC_RELEASE);
        }
    }
    void run_cpu_job()
    {
        // RAM-resident picks first (computed while the NVMe->CPU picks are still landing), then the NVMe ones
        const int ntok = cj.ntok, np = cj.np, li = cj.li;
        Layer& Ly = layers[li];
        xf.resize(size_t(std::max(ntok, 1)) * H);
        for (size_t i = 0; i < size_t(ntok) * H; ++i) xf[i] = h2f(hx[i]);
        int nb = 0; int btok[MAXP];
        if (b70r)
        {
            int32_t* rq = (int32_t*) (b70r + B70_OFF_REQ);
            uint16_t* bx = (uint16_t*) (b70r + B70_OFF_X);
            for (int k = 0; k < np; ++k)
                if (cj.lane[k] == LN_B70)
                {
                    rq[4 + nb] = cj.e[k]; ((float*) rq)[4 + MAXP + nb] = cj.w[k]; rq[4 + 2 * MAXP + nb] = cj.t[k];
                    std::memcpy(bx + size_t(nb) * H, hx + size_t(cj.t[k]) * H, size_t(H) * 2);
                    btok[nb++] = cj.t[k];
                }
            if (nb) b70_post(nb, ntok, li);
        }
        std::vector<int> pinned;
        std::vector<std::vector<std::pair<int, float>>> r1(ntok), r2(ntok);
        bool any2 = false;
        int maxm = 0; std::vector<int> cnt(E, 0);
        for (int k = 0; k < np; ++k) { maxm = std::max(maxm, ++cnt[cj.e[k]]); any2 |= cj.lane[k] == LN_NVC; }
        const int mode = cpu_mode >= 0 ? cpu_mode : (maxm <= 2 ? 2 : 0);
        auto bind = [&](int k) -> bool {   // caller holds mu; key resident
            const int e = cj.e[k], key = li * E + e;
            if (kvr[key] >= 0) { hc[HC_ERR] = 33; return false; }
            const int s = slot_of[key];
            if (std::find(pinned.begin(), pinned.end(), s) == pinned.end()) { pinc[s]++; pinned.push_back(s); }
            const int64_t a = ram_addr[s];
            Ly.ex[e].g.tr = reinterpret_cast<const uint8_t*>(a);
            Ly.ex[e].u.tr = reinterpret_cast<const uint8_t*>(a + off_u);
            Ly.ex[e].d.tr = reinterpret_cast<const uint8_t*>(a + off_d);
            return true;
        };
        {
            std::lock_guard<std::mutex> lk(mu);
            for (int k = 0; k < np; ++k)
                if (cj.lane[k] == LN_CPU)
                {
                    if (kst[li * E + cj.e[k]] != KS_RES) { hc[HC_ERR] = 34; continue; }
                    if (bind(k)) r1[cj.t[k]].push_back({ cj.e[k], cj.w[k] });
                }
        }
        bool any1 = false; for (auto& v : r1) any1 |= !v.empty();
        if (g_cpu_kern == 1 && n135::K_.merge && any2)
        {
            // N135: ONE forward over the RAM and the NVMe->CPU picks; an NVC job is bound (waits for its landing) by the
            // first worker that reaches it, after every resident job's units
            std::vector<uint8_t> late(E, 0);
            for (int k = 0; k < np; ++k) if (cj.lane[k] == LN_NVC) { late[cj.e[k]] = 1; r1[cj.t[k]].push_back({ cj.e[k], cj.w[k] }); }
            std::function<bool(int)> bind_late = [&](int ex) -> bool {
                std::unique_lock<std::mutex> lk(mu);
                int k = -1;
                for (int q = 0; q < np; ++q) if (cj.e[q] == ex && cj.lane[q] == LN_NVC) { k = q; break; }
                if (k < 0) return false;
                const int key = li * E + ex;
                if (kst[key] != KS_RES)
                {
                    const double tw = now_ms();
                    // partial: the gate+up bytes suffice to bind (the record's slot is fixed at start_read)
                    auto ok = [&] { return kst[key] == KS_RES || (partial && kst[key] == KS_INFLIGHT && gu_res[key]); };
                    land_cv.wait_for(lk, std::chrono::seconds(10), ok);
                    C.cpu_wait_land_ns += (long long) ((now_ms() - tw) * 1e6);
                    if (!ok()) { hc[HC_ERR] = 32; return false; }
                    if (kst[key] != KS_RES) C.partial_binds++;
                }
                return bind(k);
            };
            std::function<bool(int)> down_late = [&](int ex) -> bool {   // the down-projection bytes have landed
                const int key = li * E + ex;
                if (__atomic_load_n(&ram_res[key], __ATOMIC_ACQUIRE) == 1) return true;
                std::unique_lock<std::mutex> lk(mu);
                const double tw = now_ms();
                land_cv.wait_for(lk, std::chrono::seconds(10), [&] { return kst[key] == KS_RES; });
                C.cpu_wait_land_ns += (long long) ((now_ms() - tw) * 1e6);
                if (kst[key] != KS_RES) { hc[HC_ERR] = 32; return false; }
                return true;
            };
            n135::moe_forward4(pool, Ly, xf.data(), ntok, r1, hout, mode, late.data(), &bind_late, partial ? &down_late : nullptr);
            goto n135_done;
        }
        if (any1 || !nb) cpu_moe_forward(pool, Ly, xf.data(), ntok, r1, hout, mode);
        else std::memset(hout, 0, size_t(ntok) * H * 4);
        if (any2)
        {
            {
                std::unique_lock<std::mutex> lk(mu);
                for (int k = 0; k < np; ++k)
                {
                    if (cj.lane[k] != LN_NVC) continue;
                    const int key = li * E + cj.e[k];
                    if (kst[key] != KS_RES)
                    {
                        const double tw = now_ms();
                        land_cv.wait_for(lk, std::chrono::seconds(10), [&] { return kst[key] == KS_RES; });
                        C.cpu_wait_land_ns += (long long) ((now_ms() - tw) * 1e6);
                        if (kst[key] != KS_RES) { hc[HC_ERR] = 32; continue; }
                    }
                    if (bind(k)) r2[cj.t[k]].push_back({ cj.e[k], cj.w[k] });
                }
            }
            tmpout.resize(size_t(ntok) * H);
            cpu_moe_forward(pool, Ly, xf.data(), ntok, r2, tmpout.data(), mode);
            for (size_t i = 0; i < size_t(ntok) * H; ++i) hout[i] += tmpout[i];
        }
    n135_done:
        {
            std::lock_guard<std::mutex> lk(mu);
            for (int s : pinned) pinc[s]--;
        }
        if (nb) b70_collect(nb, btok);
        C.cpu_jobs++; C.cpu_experts += np - nb; C.cpu_tokens += ntok;
    }

    // ---------------- N137 B70 ring client ----------------
    volatile int64_t* b70h() { return (volatile int64_t*) b70r; }
    void b70_post(int nb, int ntok, int li)
    {
        int32_t* rq = (int32_t*) (b70r + B70_OFF_REQ);
        rq[0] = nb; rq[1] = ntok; rq[2] = li;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        b70_t0 = now_ms();
        __atomic_store_n(&b70h()[B70_W_REQ_SEQ], ++b70_seq, __ATOMIC_RELEASE);
    }
    double b70_t0 = 0;
    bool b70_wait()
    {
        const double tw = now_ms();
        long n = 0;
        while (__atomic_load_n(&b70h()[B70_W_DONE_SEQ], __ATOMIC_ACQUIRE) != b70_seq)
        {
            _mm_pause();
            if ((++n & 4095) == 0 && (now_ms() - tw) * 1e6 > b70_timeout_ns) { hc[HC_ERR] = 40; C.b70_err++; return false; }
        }
        const double t1 = now_ms();
        C.b70_wait_ns += (long long) ((t1 - tw) * 1e6);
        const long long rtt = (long long) ((t1 - b70_t0) * 1e6);
        C.b70_rtt_ns += rtt; C.b70_rtt_max_ns = std::max<long long>(C.b70_rtt_max_ns.load(), rtt);
        if (b70h()[B70_W_ERR]) { hc[HC_ERR] = 41; C.b70_err++; return false; }
        return true;
    }
    void b70_collect(int nb, const int* btok)
    {
        if (!b70_wait()) return;
        const uint16_t* bo = (const uint16_t*) (b70r + B70_OFF_OUT);
        for (int i = 0; i < nb; ++i)
        {
            float* o = hout + size_t(btok[i]) * H;
            const uint16_t* s = bo + size_t(i) * H;
            for (int j = 0; j < H; j += 8)
            {
                __m256 a = _mm256_loadu_ps(o + j);
                __m256 b = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (s + j)));
                _mm256_storeu_ps(o + j, _mm256_add_ps(a, b));
            }
        }
        C.b70_jobs++; C.b70_rows += nb;
    }

    // ---------------- stage engine (prefill) ----------------
    static void on_stage(void* ctx, int64_t tag, int err, double ms)
    {
        Engine* e = static_cast<Engine*>(ctx);
        if (err) { e->C.read_err++; e->hc[HC_ERR] = 17; }
        e->C.read_ms_x1000 += (long long) (ms * 1000);
        {
            std::lock_guard<std::mutex> lk(e->st_mu);
            e->st_nv[size_t(tag)].ready.store(1);
            e->st_inflight--;
        }
        e->st_cv.notify_all();
    }
    void st_reader()
    {
        pin_self(wb_cpu);
        const std::vector<int64_t>& RA = ring_addr();
        const int R = (int) RA.size();
        if (R <= 0) { hc[HC_ERR] = 18; return; }
        for (size_t g = 0; g < st_nv.size(); ++g)
        {
            if (st_cancel.load()) break;
            const int r = int(g % R);
            if (g >= size_t(R))
            {
                // ring entry r was used by entry g - R: wait until its H2D was enqueued, then until it completed
                std::unique_lock<std::mutex> lk(st_mu);
                st_cv.wait(lk, [&] { return st_cancel.load() || pf_ev_pending[r] == int(g - R) + 1; });
                if (st_cancel.load()) break;
                lk.unlock();
                cudaEventSynchronize(pf_ev[r]);
            }
            {
                std::unique_lock<std::mutex> lk(st_mu);
                st_cv.wait(lk, [&] { return st_cancel.load() || st_inflight.load() < st_qd; });
                if (st_cancel.load()) break;
                st_inflight++;
            }
            st_nv[g].ring = r;
            rd.submit(fd, int64_t(st_nv[g].key) * rec_bytes, RA[r], slot_bytes, piece, RC_BULK, &Engine::on_stage, this, int64_t(g), -1);
            C.stage_reads++;
        }
    }
    void stage_end()
    {
        if (!st_active) return;
        st_cancel = true;
        st_cv.notify_all();
        if (st_th.joinable()) st_th.join();
        {   // drain in-flight reads
            std::unique_lock<std::mutex> lk(st_mu);
            st_cv.wait(lk, [&] { return st_inflight.load() == 0; });
        }
        for (auto& ev : pf_ev) cudaEventSynchronize(ev);
        {
            std::lock_guard<std::mutex> lk(mu);
            for (int s : st_pins) pinc[s]--;
            for (int s : st_ring_slots) free_slot(s);
        }
        st_ring_slots.clear(); st_ring_addr.clear();
        st_pins.clear(); st_nv.clear(); st_ram_src.clear(); st_ram_j.clear(); st_nv_idx.clear();
        std::fill(pf_ev_pending.begin(), pf_ev_pending.end(), 0);
        st_active = false;
    }
    // slot_np: [L, E] int32 VRAM residency snapshot (stable during prefill: no admission)
    void stage_begin(const int32_t* so, int li0)
    {
        stage_end();
        st_cancel = false;
        st_ram_src.assign(L, {}); st_ram_j.assign(L, {}); st_nv_idx.assign(L, {});
        {
            std::lock_guard<std::mutex> lk(mu);
            // borrow the ring before classifying: a long prompt needs nearly every expert, so once its RAM-resident
            // ones are pinned there is nothing left to evict. The LRU-oldest experts evicted here are staged from
            // NVMe like any other non-resident one (free slots, e.g. a previous prefill's ring, are taken first)
            for (int i = 0; i < ring_n; ++i)
            {
                const int s = take_slot();
                if (s < 0) break;
                owner[s] = SL_STAGE; st_ring_slots.push_back(s); st_ring_addr.push_back(ram_addr[s]);
                C.ring_borrowed++;
            }
            for (int li = li0; li < L; ++li)
            {
                int j = 0;
                for (int e = 0; e < E; ++e)
                {
                    if (so[li * E + e] >= 0) continue;
                    const int key = li * E + e;
                    if (kst[key] == KS_RES)
                    {
                        const int s = slot_of[key];
                        pinc[s]++; st_pins.push_back(s);
                        st_ram_src[li].push_back(kvr[key] >= 0 ? vring_addr[kvr[key]] : ram_addr[s]); st_ram_j[li].push_back(j);
                        C.stage_ram++;
                    }
                    else
                    {
                        st_nv.emplace_back();
                        StageEntry& se = st_nv.back(); se.li = li; se.j = j; se.key = key; se.ring = -1;
                        st_nv_idx[li].push_back(int(st_nv.size()) - 1);
                        C.stage_nv++;
                    }
                    ++j;
                }
            }
        }
        st_active = true;
        st_th = std::thread([this] { st_reader(); });
    }
    // enqueue on `stream`: wait ev_wait, H2D layer li's RAM experts, then its NVMe experts as they land in the ring
    double stage_layer(int li, int64_t base, int64_t stream, int64_t ev_wait, int64_t ev_done)
    {
        const double t0 = now_ms();
        cudaStream_t st = (cudaStream_t) stream;
        cudaStreamWaitEvent(st, (cudaEvent_t) ev_wait, 0);
        double waited = 0;
        if (st_active && li < L)
        {
            for (size_t i = 0; i < st_ram_src[li].size(); ++i)
                cudaMemcpyAsync((void*) (base + st_ram_j[li][i] * slot_bytes), (const void*) st_ram_src[li][i], size_t(slot_bytes), cudaMemcpyDefault, st);
            for (int g : st_nv_idx[li])
            {
                StageEntry& se = st_nv[size_t(g)];
                if (!se.ready.load())
                {
                    const double tw = now_ms();
                    std::unique_lock<std::mutex> lk(st_mu);
                    st_cv.wait_for(lk, std::chrono::seconds(30), [&] { return se.ready.load() == 1; });
                    waited += now_ms() - tw;
                    if (!se.ready.load()) { hc[HC_ERR] = 18; continue; }
                }
                cudaMemcpyAsync((void*) (base + se.j * slot_bytes), (const void*) ring_addr()[se.ring], size_t(slot_bytes), cudaMemcpyHostToDevice, st);
                cudaEventRecord(pf_ev[se.ring], st);
                {
                    std::lock_guard<std::mutex> lk(st_mu);
                    pf_ev_pending[se.ring] = g + 1;
                }
                st_cv.notify_all();
            }
        }
        cudaEventRecord((cudaEvent_t) ev_done, st);
        C.stage_wait_ns += (long long) (waited * 1e6);
        C.stage_job_ns += (long long) ((now_ms() - t0) * 1e6);
        return waited;
    }
};

// ------------------------------------------------------------------------------------------------------------
// python API
// ------------------------------------------------------------------------------------------------------------
static struct sigaction g_old_segv;
static void segv_handler(int sig, siginfo_t* si, void* uc)
{
    void* bt[64];
    const int n = backtrace(bt, 64);
    char msg[128];
    const int m = snprintf(msg, sizeof(msg), "\nnv2: signal %d at %p, thread %ld, backtrace:\n", sig, si ? si->si_addr : nullptr, (long) pthread_self());
    if (m > 0) { ssize_t w = write(2, msg, size_t(m)); (void) w; }
    backtrace_symbols_fd(bt, n, 2);
    sigaction(SIGSEGV, &g_old_segv, nullptr);
    raise(sig);
}

void init(int64_t L, int64_t E, int64_t H, int64_t I, int64_t rec_bytes, int64_t slot_bytes, int64_t piece, int64_t off_u,
          int64_t off_d, int64_t fd, int64_t ph, int64_t hc, int64_t req, int64_t rep, int64_t wbl, int64_t adml, int64_t land,
          int64_t home, int64_t ram_res, int64_t hx, int64_t hout, at::Tensor ram_addr, at::Tensor pf_ring, at::Tensor score,
          int64_t readers, std::vector<int64_t> reader_cpus, int64_t ctl_cpu, int64_t exclusive, int64_t land_target, int64_t free_lo,
          at::Tensor vring)
{
    TORCH_CHECK(G == nullptr, "nv2 engine already initialised");
    {
        struct sigaction sa; memset(&sa, 0, sizeof(sa));
        sa.sa_sigaction = segv_handler; sa.sa_flags = SA_SIGINFO;
        sigaction(SIGSEGV, &sa, &g_old_segv);
    }
    G = new Engine();
    Engine& e = *G;
    e.L = int(L); e.E = int(E); e.K = int(L * E); e.H = int(H); e.I = int(I);
    e.rec_bytes = rec_bytes; e.slot_bytes = slot_bytes; e.piece = piece;
    { const char* v = getenv("GLM53_NV_DPIECE_KB"); e.dpiece = (v && *v ? atoll(v) : 1024) * 1024; e.dpiece = std::max<int64_t>(4096, e.dpiece / 4096 * 4096); } e.off_u = off_u; e.off_d = off_d; e.fd = int(fd); e.ph = ph;
    e.hc = (volatile int64_t*) hc; e.req = (Req*) req; e.rep = (Rep*) rep; e.wbl = (LogE*) wbl; e.adml = (LogE*) adml;
    e.land = (int32_t*) land; e.home = (volatile int64_t*) home; e.ram_res = (volatile int32_t*) ram_res;
    e.hx = (uint16_t*) hx; e.hout = (float*) hout;
    e.nram = int(ram_addr.numel());
    e.ram_addr.assign(ram_addr.data_ptr<int64_t>(), ram_addr.data_ptr<int64_t>() + e.nram);
    e.pf_ring_addr.assign(pf_ring.data_ptr<int64_t>(), pf_ring.data_ptr<int64_t>() + pf_ring.numel());
    e.pf_ev.resize(e.pf_ring_addr.size()); e.pf_ev_pending.assign(e.pf_ring_addr.size(), 0);
    for (auto& ev : e.pf_ev) cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    e.score.assign(score.data_ptr<float>(), score.data_ptr<float>() + e.K);
    e.slot_of.assign(e.K, -1); e.kvr.assign(e.K, -1); e.kst.assign(e.K, KS_NONE); e.karr.assign(e.K, 0); e.prot.assign(e.K, 0);
    e.pfs.assign(e.K, 0); e.kseq.assign(e.K, 0); e.gu_res.assign(e.K, 0);
    if (const char* pm = getenv("GLM53_NV_PARTIAL")) e.partial = atoi(pm) != 0;
    if (const char* pm = getenv("GLM53_NV_PF_MAX")) e.pf_max = std::max(0, atoi(pm));
    if (const char* pm = getenv("GLM53_NV_PF_MULTI")) e.pf_multi = std::max(0, atoi(pm));
    if (const char* pm = getenv("GLM53_NV_TNZC")) e.pol.tnzc = atof(pm);
    if (const char* pm = getenv("GLM53_NV_WB_EARLY")) e.wb_early = atoi(pm) != 0;
    if (const char* pm = getenv("GLM53_NV_RZC1")) e.rzc1 = atoi(pm) != 0;
    {
        const char* v;
        e.lossy_on = (v = getenv("GLM53_NV_LOSSY")) && atoi(v) == 1;
        if ((v = getenv("GLM53_NV_LOSSY_EPS"))) e.lossy_eps = (float) atof(v);
        if ((v = getenv("GLM53_NV_LOSSY_WHI"))) e.lossy_whi = (float) atof(v);
        if ((v = getenv("GLM53_NV_LOSSY_RENORM"))) e.lossy_renorm = atoi(v);
        if ((v = getenv("GLM53_NV_LOSSY_PF"))) e.lossy_pf = std::max(0, atoi(v));
        if ((v = getenv("GLM53_NV_LOSSY_BG"))) e.lossy_bg = std::max(0, atoi(v));
        if ((v = getenv("GLM53_NV_LOSSY_LO"))) e.lossy_lo = atoi(v);
        if ((v = getenv("GLM53_NV_LOSSY_HI"))) e.lossy_hi = atoi(v);
        if ((v = getenv("GLM53_NV_TAIL"))) e.tail_r = atoi(v);
        if ((v = getenv("GLM53_NV_TAIL_SCOPE"))) e.tail_scope = atoi(v);
        e.lwant.assign(MAXU, 0); e.luix.assign(e.E, -1);
        if (e.lossy_on) fprintf(stderr, "nv2: BCP lossy verify eps %.3f whi %.3f renorm %d pf %d bg %d layers %d..%d\n",
                                e.lossy_eps, e.lossy_whi, e.lossy_renorm, e.lossy_pf, e.lossy_bg, e.lossy_lo, e.lossy_hi);
    }
    e.owner.assign(e.nram, SL_FREE); e.pinc.assign(e.nram, 0); e.lprev.assign(e.nram, -1); e.lnext.assign(e.nram, -1);
    for (int s = e.nram - 1; s >= 0; --s) e.freel.push_back(s);
    for (int k = 0; k < e.K; ++k) { e.ram_res[k] = 0; e.set_home(k, ph); }
    e.exclusive = exclusive != 0; e.land_target = int(land_target); e.free_lo = int(free_lo);
    e.ctl_cpu = int(ctl_cpu);
    std::vector<int> rc(reader_cpus.begin(), reader_cpus.end());
    e.wb_cpu = rc.empty() ? -1 : rc.back();
    e.rd.start(int(readers), rc);
    cudaStreamCreateWithFlags(&e.wb_stream, cudaStreamNonBlocking);
    e.vring_addr.assign(vring.data_ptr<int64_t>(), vring.data_ptr<int64_t>() + vring.numel());
    e.vr_ev.resize(e.vring_addr.size());
    for (auto& ev : e.vr_ev) cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    for (int v = int(e.vring_addr.size()) - 1; v >= 0; --v) e.vfree.push_back(v);
    std::lock_guard<std::mutex> lk(e.mu);
    e.stock_landing();
}

void start()
{
    Engine& e = *G;
    e.next_seq = e.hc[HC_PUB] + 1;
    e.ctl_th = std::thread([&e] { e.ctl_loop(); });
    e.wb_th = std::thread([&e] { e.wb_loop(); });
}

void set_ring_borrow(int64_t n)
{
    Engine& e = *G;
    std::lock_guard<std::mutex> lk(e.mu);
    e.ring_n = int(std::max<int64_t>(0, n));
    const size_t R = std::max(e.pf_ring_addr.size(), size_t(e.ring_n));
    while (e.pf_ev.size() < R) { cudaEvent_t ev; cudaEventCreateWithFlags(&ev, cudaEventDisableTiming); e.pf_ev.push_back(ev); }
    e.pf_ev_pending.assign(R, 0);
}

void stg_init(at::Tensor addrs, int64_t per_layer)
{
    Engine& e = *G;
    std::lock_guard<std::mutex> lk(e.mu);
    e.stg_n = int(addrs.numel()); e.stg_max = int(per_layer);
    if (const char* sg = getenv("GLM53_NV_STG_GATE")) e.stg_gate = atoi(sg);
    e.stg_addr.assign(addrs.data_ptr<int64_t>(), addrs.data_ptr<int64_t>() + e.stg_n);
    e.stg_key.assign(e.stg_n, -1); e.stg_rslot.assign(e.stg_n, -1); e.stg_st.assign(e.stg_n, 0); e.stg_gt.assign(e.stg_n, 0);
    e.stg_use.assign(e.stg_n, -1); e.stg_after.assign(e.stg_n, 0); e.stg_t.assign(e.stg_n, 0); e.staged_of.assign(e.K, -1);
    e.stg_ev.resize(e.stg_n);
    for (auto& ev : e.stg_ev) cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    cudaStreamCreateWithFlags(&e.stg_stream, cudaStreamNonBlocking);
    e.stg_th = std::thread([&e] { e.stg_loop(); });
    fprintf(stderr, "[nv2] layer-ahead VRAM staging: %d pool slots, up to %d per layer\n", e.stg_n, e.stg_max);
}

void set_pol(std::vector<double> v)
{
    Engine& e = *G; Pol& p = e.pol;
    double* f[] = { &p.g0, &p.thit, &p.tzc, &p.ca, &p.cb, &p.ctok, &p.push, &p.nvlat, &p.nvdeep, &p.nvone };
    for (size_t i = 0; i < 10 && i < v.size(); ++i) *f[i] = v[i];
    if (v.size() > 10) p.cpu_on = int(v[10]);
    if (v.size() > 11) p.nvcpu_on = int(v[11]);
    if (v.size() > 12) p.maxcpu = int(v[12]);
    if (v.size() > 13) p.nv_noadmit = int(v[13]);
}

// blocking fill (warm start): keys -> resident (oldest-first LRU position order = given order)
int64_t warm_read(at::Tensor keys)
{
    Engine& e = *G;
    const int64_t* k = keys.data_ptr<int64_t>();
    int64_t n = 0;
    for (int64_t i = 0; i < keys.numel(); ++i)
    {
        std::lock_guard<std::mutex> lk(e.mu);
        if (e.kst[k[i]] != KS_NONE) continue;
        if (!e.start_read(int(k[i]), RC_BULK, nullptr)) break;
        ++n;
    }
    // wait for all to land
    std::unique_lock<std::mutex> lk(e.mu);
    e.land_cv.wait(lk, [&] { for (int64_t i = 0; i < keys.numel(); ++i) if (e.kst[k[i]] == KS_INFLIGHT) return false; return true; });
    return n;
}

// exclusive: drop RAM copies of these keys (VRAM-resident after the VRAM warm)
int64_t drop_keys(at::Tensor keys)
{
    Engine& e = *G;
    std::lock_guard<std::mutex> lk(e.mu);
    int64_t n = 0;
    const int64_t* k = keys.data_ptr<int64_t>();
    for (int64_t i = 0; i < keys.numel(); ++i)
        if (e.kst[k[i]] == KS_RES && e.pinc[e.slot_of[k[i]]] == 0 && e.kvr[k[i]] < 0) { e.free_slot(e.evict_key(int(k[i]))); e.C.evictions--; ++n; }
    return n;
}

// elastic enter (exclusive): VRAM-only experts of the slots being handed back -> fresh RAM slots (D2H), host-marked
int64_t wb_bulk(at::Tensor keys, at::Tensor src)
{
    Engine& e = *G;
    const int64_t* k = keys.data_ptr<int64_t>(); const int64_t* a = src.data_ptr<int64_t>();
    std::vector<std::pair<int, int>> done;
    {
        std::lock_guard<std::mutex> lk(e.mu);
        for (int64_t i = 0; i < keys.numel(); ++i)
        {
            const int key = int(k[i]);
            if (e.kst[key] != KS_NONE) continue;
            const int s = e.take_slot();
            if (s < 0) break;
            e.owner[s] = key;                        // reserved
            cudaMemcpyAsync((void*) e.ram_addr[s], (const void*) a[i], size_t(e.slot_bytes), cudaMemcpyDeviceToHost, e.wb_stream);
            done.push_back({ key, s });
        }
    }
    cudaStreamSynchronize(e.wb_stream);
    std::lock_guard<std::mutex> lk(e.mu);
    for (auto& [key, s] : done)
    {
        e.slot_of[key] = s; e.kst[key] = KS_RES; e.lru_push_new(s); e.set_home(key, e.ram_addr[s]);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        __atomic_store_n(&e.ram_res[key], 1, __ATOMIC_RELEASE);
    }
    e.C.wb_bulk += (long long) done.size();
    return (int64_t) done.size();
}

// sync drain of device logs (call after torch.cuda.synchronize, e.g. before verify / elastic)
void drain_all()
{
    Engine& e = *G;
    {
        std::lock_guard<std::mutex> lk(e.mu);
        e.drain_logs(INT64_MAX);
    }
    e.wb_cv.notify_all();
    for (int i = 0; i < 20000; ++i)
    {
        bool empty;
        { std::lock_guard<std::mutex> lw(e.wb_mu); empty = e.wb_q.empty() && e.wb_inflight.load() == 0; }
        if (empty) break;
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
}

// key state snapshot: [K] slot (or -1), [K] state
std::vector<at::Tensor> state()
{
    Engine& e = *G;
    std::lock_guard<std::mutex> lk(e.mu);
    auto so = torch::from_blob(e.slot_of.data(), { e.K }, torch::kInt32).clone();
    auto ks = torch::from_blob(e.kst.data(), { e.K }, torch::kInt8).clone();
    auto pc = torch::from_blob(e.pinc.data(), { e.nram }, torch::kInt32).clone();
    auto ow = torch::from_blob(e.owner.data(), { e.nram }, torch::kInt32).clone();
    return { so, ks, pc, ow };
}

int64_t read_sync(int64_t key, int64_t dst)
{
    Engine& e = *G;
    int64_t done = 0;
    while (done < e.slot_bytes)
    {
        ssize_t r = pread(e.fd, (void*) (dst + done), size_t(e.slot_bytes - done), off_t(key * e.rec_bytes + done));
        if (r <= 0) { if (r < 0 && errno == EINTR) continue; return -1; }
        done += r;
    }
    return done;
}

std::vector<int64_t> counters()
{
    Engine& e = *G; Counters& c = e.C;
    std::vector<int64_t> v = { c.reqs, c.dec_reqs, c.small_reqs, c.picks, c.vram, c.ram, c.nvme, c.inflight_hit, c.lane_zc,
        c.lane_cpu, c.lane_nvg, c.lane_nvc, c.reads, c.read_bytes, c.wb_landed, c.wb_dup, c.adm_freed, c.adm_pinned,
        c.evictions, c.cpu_jobs, c.cpu_experts, c.cpu_tokens, c.stage_ram, c.stage_nv, c.stage_reads, c.read_err,
        c.alloc_fail, c.land_pushed, c.prefetch, c.plan_ns, c.ctl_busy_ns, c.cpu_busy_ns, c.cpu_wait_land_ns,
        c.read_ms_x1000, c.stage_wait_ns, c.stage_job_ns, c.wb_bulk, c.dec_tok, c.notice_ns, c.notice_max_ns, c.serve_max_ns, c.reply_ns, c.wb_issued, c.wb_drop, c.vring_hits,
        c.lane_b70, c.b70_jobs, c.b70_rows, c.b70_wait_ns, c.b70_rtt_ns, c.b70_err, c.b70_rtt_max_ns, c.pf_keys, c.pf_skip, c.pf_used, c.pf_wasted, c.lane_skip, c.lossy_calls, c.lossy_tok, c.lossy_tok_skip,
        c.lossy_cold_x1e6, c.lossy_skip_x1e6, c.lossy_coldu, c.lossy_want, c.lossy_bg_reads, c.promoted,
        c.stg_issued, c.stg_ready, c.stg_hit, c.stg_late, c.stg_nofree, c.stg_dup, c.stg_waste, c.wb_early,
        c.partial_lands, c.partial_binds, c.wb_lat_us, c.wb_lat_max_us, c.ring_low, c.ring_low_vfree, c.ring_borrowed };
    std::lock_guard<std::mutex> lk(e.mu);
    int res = 0, inf = 0; for (int k = 0; k < e.K; ++k) { res += e.kst[k] == KS_RES; inf += e.kst[k] == KS_INFLIGHT; }
    v.push_back(res); v.push_back(inf); v.push_back((int64_t) e.freel.size()); v.push_back(e.land_tail - e.hc[HC_WBN]);
    return v;
}

std::vector<at::Tensor> dbg_host()
{
    Engine& e = *G;
    auto T = [](std::vector<long long>& v) { return torch::from_blob(v.data(), { DBG }, torch::kInt64).clone(); };
    return { T(e.dbg_seq), T(e.dbg_tn), T(e.dbg_tr), T(e.dbg_cs), T(e.dbg_ce), T(e.dbg_nr), T(e.dbg_npf), T(e.dbg_inf), T(e.dbg_lm) };
}

void trace_set(int64_t on) { Engine& e = *G; std::lock_guard<std::mutex> lt(e.trace_mu); e.trace_on = on != 0; if (!on) e.trace.clear(); }
at::Tensor trace_take()
{
    Engine& e = *G;
    std::lock_guard<std::mutex> lt(e.trace_mu);
    auto t = torch::from_blob(e.trace.data(), { (int64_t) e.trace.size() }, torch::kInt32).clone();
    e.trace.clear();
    return t;
}

std::vector<int64_t> diag()
{
    Engine& e = *G;
    std::vector<int64_t> v;
    for (int i = 0; i < 8; ++i) v.push_back((int64_t) e.hc[i]);
    v.push_back(e.next_seq); v.push_back((int64_t) e.wb_inflight.load());
    v.push_back((int64_t) e.wb_q.size());
    v.push_back((int64_t) e.vfree.size()); v.push_back(e.land_tail); v.push_back(e.wb_read); v.push_back(e.adm_read);
    const Req& r = e.req[e.next_seq % RQ];
    v.push_back(r.seq);
    return v;
}

void reset_counters() { Counters& c = G->C; c.~Counters(); new (&c) Counters(); }

// ---- CPU tier ----
void cpu_init(int64_t threads, std::vector<int64_t> cpus, int64_t mode, double act_limit)
{
    Engine& e = *G;
    init_perm(); init_tables(); cpu_kern_init();
    std::vector<int> c(cpus.begin(), cpus.end());
    cpu_set_t saved; sched_getaffinity(0, sizeof(saved), &saved);
    e.pool.start(int(threads), c);
    sched_setaffinity(0, sizeof(saved), &saved);
    e.cpu_mode = int(mode);
    g_act_limit = float(act_limit);
    e.layers.resize(e.L);
}
void cpu_add_layer(int64_t li, at::Tensor suh_g, at::Tensor svh_g, at::Tensor suh_u, at::Tensor svh_u, at::Tensor suh_d, at::Tensor svh_d)
{
    Engine& e = *G;
    Layer& Ly = e.layers[li];
    Ly.H = e.H; Ly.I = e.I; Ly.ex.resize(e.E);
    auto u16 = [](const at::Tensor& t, int x) { return reinterpret_cast<const uint16_t*>(t.data_ptr()) + size_t(x) * t.size(1); };
    for (int x = 0; x < e.E; ++x)
    {
        Ly.ex[x].g = { nullptr, u16(suh_g, x), u16(svh_g, x), e.H, e.I };
        Ly.ex[x].u = { nullptr, u16(suh_u, x), u16(svh_u, x), e.H, e.I };
        Ly.ex[x].d = { nullptr, u16(suh_d, x), u16(svh_d, x), e.I, e.H };
    }
    for (auto& t : { suh_g, svh_g, suh_u, svh_u, suh_d, svh_d }) e.keep.push_back(t);
}
void cpu_start(int64_t cpu0)
{
    Engine& e = *G;
    e.cpu_ready = true;
    e.cpu_th = std::thread([&e, cpu0] { e.cpu_loop(int(cpu0)); });
}
void cpu_set(int64_t on, int64_t nvcpu, int64_t mode, double act_limit)
{
    Engine& e = *G;
    e.pol.cpu_on = int(on); e.pol.nvcpu_on = int(nvcpu); e.cpu_mode = int(mode); g_act_limit = float(act_limit);
}
// synchronous forward from RAM slots (worker idle / CPU lane off): x fp32 [m, H], sel int32 [m, k], w fp32 [m, k]
at::Tensor cpu_forward(int64_t li, at::Tensor x, at::Tensor sel, at::Tensor w, int64_t mode)
{
    Engine& e = *G;
    const int m = int(x.size(0)), k = int(sel.size(1));
    auto out = torch::zeros({ m, e.H }, torch::kFloat);
    std::vector<std::vector<std::pair<int, float>>> route(m);
    auto sa = sel.accessor<int32_t, 2>(); auto wa = w.accessor<float, 2>();
    Layer& Ly = e.layers[li];
    std::vector<int> pinned;
    {
        std::lock_guard<std::mutex> lk(e.mu);
        for (int t = 0; t < m; ++t)
            for (int j = 0; j < k; ++j)
            {
                const int ex = sa[t][j]; const int key = int(li) * e.E + ex;
                TORCH_CHECK(e.kst[key] == KS_RES, "cpu_forward: expert not RAM-resident");
                const int s = e.slot_of[key];
                e.pinc[s]++; pinned.push_back(s);
                const int64_t a = e.ram_addr[s];
                Ly.ex[ex].g.tr = (const uint8_t*) a; Ly.ex[ex].u.tr = (const uint8_t*) (a + e.off_u); Ly.ex[ex].d.tr = (const uint8_t*) (a + e.off_d);
                route[t].push_back({ ex, wa[t][j] });
            }
    }
    int maxm = 0; { std::vector<int> cnt(e.E, 0); for (auto& r : route) for (auto& pr : r) maxm = std::max(maxm, ++cnt[pr.first]); }
    const int md = mode >= 0 ? int(mode) : (maxm <= 2 ? 2 : 0);
    cpu_moe_forward(e.pool, Ly, x.data_ptr<float>(), m, route, out.data_ptr<float>(), md);
    std::lock_guard<std::mutex> lk(e.mu);
    for (int s : pinned) e.pinc[s]--;
    return out;
}

// ---- N137 B70 tier ----
// test-only engine (no CUDA, no readers): geometry only, so b70_attach / b70_forward can be checked without a GPU
void b70_test_engine(int64_t L, int64_t E, int64_t H)
{
    TORCH_CHECK(G == nullptr, "nv2 engine already initialised");
    G = new Engine();
    G->L = int(L); G->E = int(E); G->K = int(L * E); G->H = int(H);
    static int64_t hc_dummy[HC_N] = {0};
    G->hc = (volatile int64_t*) hc_dummy;
}
// attach the ring, LOAD the key set on the server (blocks until it answers), mark the keys B70-resident
int64_t b70_attach(std::string path, at::Tensor keys, double load_timeout_s)
{
    Engine& e = *G;
    const int fd = open(path.c_str(), O_RDWR);
    TORCH_CHECK(fd >= 0, "b70: cannot open ring ", path);
    void* p = mmap(nullptr, B70_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    TORCH_CHECK(p != MAP_FAILED, "b70: mmap failed");
    e.b70r = (uint8_t*) p;
    volatile int64_t* h = e.b70h();
    TORCH_CHECK(h[B70_W_MAGIC] == B70_MAGIC, "b70: bad ring magic");
    const int n = int(keys.numel());
    TORCH_CHECK(n <= 16384, "b70: too many keys");
    int32_t* kk = (int32_t*) (e.b70r + B70_OFF_KEYS);
    auto ka = keys.accessor<int64_t, 1>();
    for (int i = 0; i < n; ++i) kk[i] = int32_t(ka[i]);
    h[B70_W_NKEYS] = n;
    const int64_t ls = h[B70_W_LOAD_SEQ] + 1;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    h[B70_W_LOAD_SEQ] = ls;
    const double t0 = now_ms();
    while (h[B70_W_LOAD_ACK] != ls)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        TORCH_CHECK((now_ms() - t0) / 1e3 < load_timeout_s, "b70: LOAD timed out");
    }
    TORCH_CHECK(h[B70_W_STATE] == 2, "b70: server LOAD failed, err ", (long long) h[B70_W_ERR]);
    e.b70k.assign(e.K, 0);
    for (int i = 0; i < n; ++i) e.b70k[ka[i]] = 1;
    e.b70_seq = h[B70_W_DONE_SEQ];
    return (int64_t) ((now_ms() - t0));
}
void b70_set(int64_t on, double timeout_s) { Engine& e = *G; TORCH_CHECK(e.b70r || !on, "b70: not attached"); e.b70_on = on != 0; e.b70_timeout_ns = int64_t(timeout_s * 1e9); }
// synchronous forward through the ring (decode idle or B70 lane off): x fp32 [m, H], sel int32 [m, k], w fp32 [m, k]
at::Tensor b70_forward(int64_t li, at::Tensor x, at::Tensor sel, at::Tensor w)
{
    Engine& e = *G;
    TORCH_CHECK(e.b70r, "b70: not attached");
    const int m = int(x.size(0)), k = int(sel.size(1));
    TORCH_CHECK(m * k <= MAXP, "b70_forward: too many picks");
    auto sa = sel.accessor<int32_t, 2>(); auto wa = w.accessor<float, 2>(); auto xa = x.accessor<float, 2>();
    int32_t* rq = (int32_t*) (e.b70r + B70_OFF_REQ);
    uint16_t* bx = (uint16_t*) (e.b70r + B70_OFF_X);
    int nb = 0; int btok[MAXP];
    for (int t = 0; t < m; ++t)
        for (int j = 0; j < k; ++j)
        {
            TORCH_CHECK(e.b70k[li * e.E + sa[t][j]], "b70_forward: expert not B70-resident");
            rq[4 + nb] = sa[t][j]; ((float*) rq)[4 + MAXP + nb] = wa[t][j]; rq[4 + 2 * MAXP + nb] = t;
            for (int c = 0; c < e.H; ++c) bx[size_t(nb) * e.H + c] = f2h(xa[t][c]);
            btok[nb++] = t;
        }
    auto out = torch::zeros({ m, e.H }, torch::kFloat);
    float* save = e.hout;
    e.hout = out.data_ptr<float>();
    e.b70_post(nb, m, int(li));
    e.b70_collect(nb, btok);
    e.hout = save;
    return out;
}

// ---- stage engine ----
void stage_begin(at::Tensor so, int64_t li0) { G->stage_begin(so.data_ptr<int32_t>(), int(li0)); }
double stage_layer(int64_t li, int64_t base, int64_t stream, int64_t ev_wait, int64_t ev_done) { return G->stage_layer(int(li), base, stream, ev_wait, ev_done); }
void stage_end() { G->stage_end(); }
void set_stage_qd(int64_t q) { G->st_qd = int(q); }

void shutdown()
{
    if (!G) return;
    G->stage_end();
    G->quit = true;
    if (G->ctl_th.joinable()) G->ctl_th.join();
    G->wb_cv.notify_all();
    if (G->wb_th.joinable()) G->wb_th.join();
    if (G->cpu_th.joinable()) G->cpu_th.join();
    G->stg_cv.notify_all();
    if (G->stg_th.joinable()) G->stg_th.join();
    if (G->cpu_ready) G->pool.stop();
    G->rd.stop();
}

}  // namespace nv2h

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    using namespace nv2h;
    namespace py = pybind11;
    m.def("init", &init);
    m.def("start", &start);
    m.def("set_pol", &set_pol);
    m.def("stg_init", &stg_init);
    m.def("set_ring_borrow", &set_ring_borrow);
    m.def("warm_read", &warm_read, py::call_guard<py::gil_scoped_release>());
    m.def("drop_keys", &drop_keys);
    m.def("wb_bulk", &wb_bulk, py::call_guard<py::gil_scoped_release>());
    m.def("drain_all", &drain_all);
    m.def("state", &state);
    m.def("read_sync", &read_sync, py::call_guard<py::gil_scoped_release>());
    m.def("counters", &counters);
    m.def("reset_counters", &reset_counters);
    m.def("dbg_host", &dbg_host);
    m.def("trace_set", &trace_set);
    m.def("set_read_rate", [](double mbps) {   // emulate a slower NVMe device (MB/s, 0 = off): capacity planning
        Engine& e = *G; std::lock_guard<std::mutex> lk(e.rd.mu); e.rd.rate_bps = std::max(0.0, mbps) * 1e6; e.rd.vfin = 0;
    });
    m.def("cpu_knob", [](std::string k, int v) {   // n135 CPU-lane knobs at runtime (bench / A-B); returns the old value
        cpu_kern_init();
        if (k == "vbmi") { const int old = n135::vbmi_ok; if (v >= 0) n135::vbmi_ok = v && __builtin_cpu_supports("avx512vbmi"); return old; }
        int* p = k == "rt" ? &n135::K_.rt : k == "kern" ? &n135::K_.kern : k == "pfr" ? &n135::K_.pfr : k == "rg" ? &n135::K_.rg :
                 k == "cg" ? &n135::K_.cg : k == "rd" ? &n135::K_.rd : k == "cd" ? &n135::K_.cd : k == "bandmaxm" ? &n135::K_.bandmaxm :
                 k == "merge" ? &n135::K_.merge : nullptr;
        TORCH_CHECK(p, "unknown knob ", k);
        const int old = *p; if (v >= 0) *p = v; return old;
    });
    m.def("trace_take", &trace_take);
    m.def("diag", &diag);
    m.def("cpu_init", &cpu_init);
    m.def("cpu_add_layer", &cpu_add_layer);
    m.def("cpu_start", &cpu_start);
    m.def("cpu_set", &cpu_set);
    m.def("cpu_forward", &cpu_forward, py::call_guard<py::gil_scoped_release>());
    m.def("b70_attach", &b70_attach, py::call_guard<py::gil_scoped_release>());
    m.def("b70_set", &b70_set);
    m.def("b70_test_engine", &b70_test_engine);
    m.def("b70_forward", &b70_forward, py::call_guard<py::gil_scoped_release>());
    m.def("stage_begin", &stage_begin, py::call_guard<py::gil_scoped_release>());
    m.def("stage_layer", &stage_layer, py::call_guard<py::gil_scoped_release>());
    m.def("stage_end", &stage_end, py::call_guard<py::gil_scoped_release>());
    m.def("set_stage_qd", &set_stage_qd);
    m.def("shutdown", &shutdown, py::call_guard<py::gil_scoped_release>());
    m.def("set_lossy", [](int on, double eps, double whi, int renorm, int pf, int bg, int lo, int hi) {
        Engine& e = *G; std::lock_guard<std::mutex> lk(e.mu);
        e.lossy_on = on == 1; e.lossy_eps = (float) eps; e.lossy_whi = (float) whi; e.lossy_renorm = renorm;
        e.lossy_pf = std::max(0, pf); e.lossy_bg = std::max(0, bg); e.lossy_lo = lo; e.lossy_hi = hi;
        if (e.luix.size() != size_t(e.E)) e.luix.assign(e.E, -1);
        if (e.lwant.size() != size_t(MAXU)) e.lwant.assign(MAXU, 0);
    });
    m.def("set_tail", [](int r, int scope) {
        Engine& e = *G; std::lock_guard<std::mutex> lk(e.mu);
        e.tail_r = std::max(0, std::min(r, 9)); e.tail_scope = scope == 1 ? 1 : 0;
    });
    m.def("sizes", [] { return std::vector<int64_t>{ (int64_t) sizeof(Req), (int64_t) sizeof(Rep), (int64_t) sizeof(LogE), MAXU, MAXP, RQ, LG, LR, MAXB, HC_N, DC_N, ST_N, DBG, DBGW }; });
    dev_bind(m);
}

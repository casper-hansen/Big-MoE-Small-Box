"""Prompt and follow-up checkpoints for the recurrent (GDN / KDA) layers of GLM-5.3, as two patches to the installed
exllamav3 1.5.1 (generator/job.py and modules/gated_delta_net.py). Applied in place, once per install.

- Single-pass prefill checkpoint: a long prompt is prefilled across the last 256-token page boundary in one pass, not
  two. The layer snapshots its recurrent state at the boundary, and the job stashes the snapshot under the boundary
  page's hash. Saves one NVMe pass over the experts per long prompt (about 9 s on our box).
- End-of-job tail checkpoint: a finished job keeps its recurrent state and its partial last page, keyed by the exact
  token ids [0, E). A follow-up whose prompt starts with those ids restores both and prefills only the new tokens.
  Up to four are kept.

Default on. GLM53_EXL3_PATCHES=0 leaves exllamav3 stock. The patch only touches files whose sha256 matches the pinned
stock 1.5.1 files; anything else is left alone with a warning. A marker line makes a second apply a no-op. Each patched
file is written only after it compiles.
"""
import hashlib, importlib.util, os, sys

MARKER = "# bigmoe-glm53: exllamav3 patch v1 (single-pass prefill checkpoint, end-of-job tail checkpoint)"
STOCK_SHA256 = {
    "generator/job.py": "4564b54b589fa045c9452f4e5e39cb304a090e06714f6f0d2428cbb261eab04b",
    "modules/gated_delta_net.py": "11fc7f7748b97648bc282fb6cd47bac049185aebaef0475c7b81184b9d00a753",
}


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _edit_gdn(s):
    """Single-pass: split the recurrence at the checkpoint boundary inside the torch prefill path."""
    conv = "        # Convolution\n        mixed_qkv = causal_conv1d_update(\n"
    assert s.count(conv) == 1
    start = s.index(conv)
    end = s.index("        # Norm\n        core_attn_out = self.norm.forward(core_attn_out, params, gate = z)")
    orig = s[start:end]
    new = '''        # Convolution + delta rule. Single-pass checkpoint: with recurrent_checkpoint_at = b the recurrence runs as
        # [0, b) then [b, L), and the state after b is snapshotted for the job's prefix checkpoint.
        ck = params.get("recurrent_checkpoint_at")
        if ck and save_state and not save_history and recurrent_state is not None and bsz == 1 and 0 < ck < seqlen:
            cdim = self.conv_kernel_size

            def _seg(a, b_):
                mq = causal_conv1d_update(
                    mixed_qkv = mixed_qkv[:, :, a:b_].contiguous(),
                    conv_state = conv_state,
                    recurrent_slots = recurrent_slots,
                    conv1d_weight = self.conv1d_weight_flat,
                    conv1d_bias = self.conv1d_bias,
                    history = False,
                    params = params,
                )
                return gated_delta_rule_fn(
                    mixed_qkv = mq,
                    beta = beta[:, a:b_].contiguous(),
                    g = g[:, a:b_].contiguous(),
                    recurrent_state = recurrent_state,
                    recurrent_slots = recurrent_slots,
                    history = False,
                    save_state = True,
                    num_k_heads = self.num_k_heads,
                    num_v_heads = self.num_v_heads,
                    k_dim = self.k_dim,
                    v_dim = self.v_dim,
                    k_head_dim = self.k_head_dim,
                    v_head_dim = self.v_head_dim,
                    params = params,
                    channelwise_g = self.kda,
                )

            o1 = _seg(0, ck)
            slot = int(recurrent_slots[0])
            rsl.ckpt_snap = getattr(rsl, "ckpt_snap", {})
            rsl.ckpt_snap[slot] = (
                recurrent_state[slot, :1].clone(),
                conv_state[slot, :, :cdim].clone(),
            )
            o2 = _seg(ck, seqlen)
            core_attn_out = torch.cat([o1, o2], dim = 1)
        else:
''' + "\n".join(("    " + ln if ln.strip() else ln) for ln in orig.rstrip("\n").split("\n")) + "\n\n"
    s = s[:start] + new + s[end:]
    s = _rep(s, "    def tp_export(self, plan):\n",
             "    def take_ckpt(self, slot):\n"
             "        \"\"\"Snapshot taken inside a single-pass prefill at the checkpoint boundary, in the stash layout.\"\"\"\n"
             "        rec, conv = self.ckpt_snap.pop(slot)\n"
             "        return (rec.cpu(), conv.cpu())\n\n"
             "    def tp_export(self, plan):\n")
    return s


def _edit_job(s):
    """Single-pass and tail checkpoints in the prefill loop, the page release path, and two helpers each."""
    s = _rep(s, "            recurrent_last_page = False\n"
                "            if self.generator.recurrent_cache is not None:\n"
                "                seqlen = len(seq.sequence_ids) - 1\n"
                "                last_page_b = seqlen // PAGE_SIZE * PAGE_SIZE\n"
                "                if prefill_start < last_page_b <= prefill_end:\n"
                "                    prefill_end = last_page_b\n"
                "                    recurrent_last_page = True\n"
                "                    prefill_ids = seq.sequence_ids.torch_slice(prefill_start, prefill_end)\n",
             "            recurrent_last_page = False\n"
             "            ckpt_rel = None\n"
             "            ckpt_pos = None\n"
             "            if self.generator.recurrent_cache is not None:\n"
             "                seqlen = len(seq.sequence_ids) - 1\n"
             "                last_page_b = seqlen // PAGE_SIZE * PAGE_SIZE\n"
             "                if prefill_start < last_page_b <= prefill_end:\n"
             "                    if self._single_pass_ok(prefill_start, last_page_b, prefill_end):\n"
             "                        ckpt_rel = last_page_b - prefill_start\n"
             "                        ckpt_pos = last_page_b\n"
             "                    else:\n"
             "                        prefill_end = last_page_b\n"
             "                        recurrent_last_page = True\n"
             "                        prefill_ids = seq.sequence_ids.torch_slice(prefill_start, prefill_end)\n")
    s = _rep(s, '                    "mm_span_prefix": mm_span_prefix,\n                }\n',
             '                    "mm_span_prefix": mm_span_prefix,\n                }\n'
             '                if ckpt_rel is not None:\n'
             '                    params["recurrent_checkpoint_at"] = ckpt_rel\n')
    s = _rep(s, "                if recurrent_last_page:\n"
                "                    self.maybe_stash_recurrent(self.generator.recurrent_cache, PAGE_SIZE)\n",
             "                if recurrent_last_page:\n"
             "                    self.maybe_stash_recurrent(self.generator.recurrent_cache, PAGE_SIZE)\n"
             "                if ckpt_rel is not None:\n"
             "                    self._stash_inner_checkpoint(self.generator.recurrent_cache, ckpt_pos)\n")
    s = _rep(s, "            cp_pos = self.recurrent_state.position if self.recurrent_state is not None else -1\n",
             "            if self._tail_restore(seq):\n"
             "                pass\n"
             "            cp_pos = self.recurrent_state.position if self.recurrent_state is not None else -1\n")
    s = _rep(s, "            assert prefill_start % PAGE_SIZE == 0 or mm_exact_chunks\n",
             "            assert prefill_start % PAGE_SIZE == 0 or mm_exact_chunks or prefill_start == getattr(self, '_tail_E', -1)\n")
    s = _rep(s, "    def maybe_stash_recurrent(self, cache, interval = None):\n",
             "    def _single_pass_ok(self, start, boundary, end):\n"
             "        if not (__import__('os').environ.get('GLM53_EXL3_PATCHES', '1') == '1'):\n"
             "            return False\n"
             "        if self.generator.draft_model is not None or self.generator.mtp_draft or self.embeddings:\n"
             "            return False\n"
             "        if not (boundary < end):\n"
             "            return False\n"
             "        return (boundary - start) > 0 and (boundary - start) % 64 == 0\n\n"
             "    def _stash_inner_checkpoint(self, cache, pos):\n"
             "        seq = self.sequences[0]\n"
             "        rs = self.recurrent_state\n"
             "        page = seq.allocated_pages[pos // PAGE_SIZE - 1]\n"
             "        assert page.kv_position == PAGE_SIZE\n"
             "        layers = rs.cache.get_all_recurrent_layers()\n"
             "        snap = {k: l.take_ckpt(rs.slot) for k, l in layers.items()}\n"
             "        size = rs.checkpoint_size\n"
             "        class _Snap:\n"
             "            def stash(self_inner):\n"
             "                d = {'position': pos, 'checkpoint_size': size}\n"
             "                d.update(snap)\n"
             "                return d\n"
             "        cache.put(page.phash, _Snap())\n"
             "        self.last_recurrent_checkpoint_pos = pos\n\n"
             "    def maybe_stash_recurrent(self, cache, interval = None):\n")
    s = _rep(s, "    def deallocate_pages(self):\n        self.free_recurrent_state()\n",
             "    def _tail_enabled(self):\n"
             "        return (__import__('os').environ.get('GLM53_EXL3_PATCHES', '1') == '1' and self.generator.recurrent_cache is not None\n"
             "                and self.generator.draft_model is None and not self.generator.mtp_draft\n"
             "                and not self.embeddings and len(self.sequences) == 1)\n\n"
             "    def _tail_store(self):\n"
             "        if not self._tail_enabled() or self.recurrent_state is None or not self.sequences[0].allocated_pages:\n"
             "            return\n"
             "        seq = self.sequences[0]\n"
             "        E = seq.kv_position\n"
             "        if E <= 0 or E % PAGE_SIZE == 0 or self.recurrent_state.position != E:\n"
             "            return\n"
             "        boundary = E // PAGE_SIZE * PAGE_SIZE\n"
             "        n = E - boundary\n"
             "        src = seq.allocated_pages[E // PAGE_SIZE]\n"
             "        if src.kv_position < n:\n"
             "            return\n"
             "        store = getattr(self.generator, 'tail_ckpts', None)\n"
             "        if store is None:\n"
             "            store = []\n"
             "            self.generator.tail_ckpts = store\n"
             "        pt = self.pagetable\n"
             "        cand = None\n"
             "        for p in pt.unreferenced_pages.values():\n"
             "            if p.ref_count == 0 and p.kv_position == 0:\n"
             "                cand = p\n"
             "                break\n"
             "        if cand is None:\n"
             "            return\n"
             "        while len(store) >= 4:\n"
             "            store.pop(0)['page'].sub_ref()\n"
             "        from .pagetable import _randomhash\n"
             "        cand.add_ref_clear(0, _randomhash())\n"
             "        self.generator.cache.copy_page(self.generator.cache, src.page_index, cand.page_index, n)\n"
             "        store.append({'E': E, 'n': n, 'page': cand,\n"
             "                      'ids': seq.sequence_ids.torch_slice(0, E).cpu().clone(),\n"
             "                      'stash': self.recurrent_state.stash()})\n\n"
             "    def _tail_restore(self, seq):\n"
             "        if not self._tail_enabled():\n"
             "            return False\n"
             "        store = getattr(self.generator, 'tail_ckpts', None) or []\n"
             "        boundary = seq.kv_position\n"
             "        if boundary % PAGE_SIZE != 0:\n"
             "            return False\n"
             "        for ent in store:\n"
             "            E = ent['E']\n"
             "            if E // PAGE_SIZE * PAGE_SIZE != boundary or E >= len(seq.sequence_ids) - 1:\n"
             "                continue\n"
             "            if not bool((seq.sequence_ids.torch_slice(0, E).cpu() == ent['ids']).all()):\n"
             "                continue\n"
             "            page = seq.allocated_pages[E // PAGE_SIZE]\n"
             "            if page.kv_position != 0:\n"
             "                return False\n"
             "            n = ent['n']\n"
             "            self.generator.cache.copy_page(self.generator.cache, ent['page'].page_index, page.page_index, n)\n"
             "            page.sequence[:, :n].copy_(seq.sequence_ids.torch_slice(boundary, E))\n"
             "            page.prev_hash = seq.allocated_pages[E // PAGE_SIZE - 1].phash if E // PAGE_SIZE > 0 else None\n"
             "            page.kv_position = n\n"
             "            page.can_revert = False\n"
             "            seq.kv_position = E\n"
             "            self.cached_tokens += n\n"
             "            self.free_recurrent_state()\n"
             "            self.recurrent_state = self.generator.cache.new_from_stashed(ent['stash'], position = E)\n"
             "            self.last_recurrent_checkpoint_pos = E\n"
             "            self._tail_E = E\n"
             "            return True\n"
             "        return False\n\n"
             "    def deallocate_pages(self):\n"
             "        self._tail_store()\n"
             "        self.free_recurrent_state()\n")
    return s


def _rep(s, old, new):
    n = s.count(old)
    if n != 1:
        raise RuntimeError("exllamav3 patch anchor found %d times: %r" % (n, old[:80]))
    return s.replace(old, new)


def _package_root():
    spec = importlib.util.find_spec("exllamav3")
    if spec is None or not spec.submodule_search_locations:
        return None
    return list(spec.submodule_search_locations)[0]


def apply():
    """Patch the installed exllamav3 in place. Returns a short status string; never raises for a non-stock install."""
    if os.environ.get("GLM53_EXL3_PATCHES", "1") != "1":
        return "disabled (GLM53_EXL3_PATCHES=0)"
    root = _package_root()
    if root is None:
        return "exllamav3 not installed; nothing patched"
    texts = {}
    for rel in STOCK_SHA256:
        path = os.path.join(root, rel)
        with open(path) as f:
            texts[rel] = f.read()
    if all(MARKER in t for t in texts.values()):
        return "already patched"
    if any(MARKER in t for t in texts.values()) or any(_sha(t) != STOCK_SHA256[r] for r, t in texts.items()):
        print(" !! exllamav3 files are not the pinned stock 1.5.1 files; prompt/tail checkpoint patch not applied",
              file=sys.stderr, flush=True)
        return "skipped (not stock 1.5.1)"
    new = {
        "generator/job.py": MARKER + "\n" + _edit_job(texts["generator/job.py"]),
        "modules/gated_delta_net.py": MARKER + "\n" + _edit_gdn(texts["modules/gated_delta_net.py"]),
    }
    for rel, text in new.items():           # compile everything before writing anything
        compile(text, rel, "exec")
    for rel, text in new.items():
        path = os.path.join(root, rel)
        bak = path + ".glm53-stock"
        if not os.path.exists(bak):
            with open(bak, "w") as f:
                f.write(texts[rel])
        tmp = path + ".glm53-tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    return "applied"

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
    RpaCase,
    get_default_block_sizes,
)
from tpu_inference.kernels.experimental.batched_rpa.wrapper import (
    ragged_paged_attention,
)

from benchmark.utils import multiple_iteration_timeit_from_trace
from utils import create_decode_uniform_data, create_prefill_uniform_data


@functools.partial(
    jax.jit,
    static_argnames=["sm_scale", "sliding_window"],
    donate_argnames=["q", "kv_cache"],
)
def _jitted_attn(
    q,
    k,
    v,
    kv_cache,
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
    sm_scale,
    sliding_window=None,
):
    return ragged_paged_attention(
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
    )


def _run_attention_benchmark(
    data_factory,
    rpa_case,
    q_head_num,
    kv_head_num,
    head_dim,
    page_size,
    max_num_batched_tokens,
    sliding_window=None,
):
    (
        q,
        k,
        v,
        _kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        _,
        _,
        _,
        _,
        distribution,
    ) = data_factory()

    scale = head_dim**-0.5
    max_num_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    block_sizes = get_default_block_sizes(
        q.dtype,
        k.dtype,
        q_head_num,
        kv_head_num,
        head_dim,
        page_size,
        max_num_batched_tokens,
        max_num_seqs,
        pages_per_seq,
        case=rpa_case,
    )

    def _gen_qkv_cache():
        sample = data_factory()
        q_, k_, v_, kv_cache_ = sample[0], sample[1], sample[2], sample[3]
        jax.block_until_ready((q_, k_, v_, kv_cache_))
        return (q_, k_, v_, kv_cache_)

    def _compute(q_, k_, v_, kv_cache_):
        return _jitted_attn(
            q_,
            k_,
            v_,
            kv_cache_,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            scale,
            sliding_window=sliding_window,
        )

    scope_name = (
        f"RPA{rpa_case.symbol}-p_{page_size}"
        f"-bq_{block_sizes["bq_sz"]}_{block_sizes["bq_csz"]}"
        f"-bkv_{block_sizes["bkv_sz"]}_{block_sizes["bkv_csz"]}"
    )
    if sliding_window is not None:
        scope_name += f"-sw_{sliding_window}"

    scope_name = "jit__jitted_attn*"

    times = multiple_iteration_timeit_from_trace(
        compute_func=_compute,
        data_generator=_gen_qkv_cache,
        task=scope_name,
        tries=5,
        warmup=3,
    )
    return float(np.mean(times)) if times else float("nan")


def benchmark_prefill_backend(
    max_context_len,
    max_kv_cache_tokens,
    max_num_batched_tokens,
    q_head_num,
    kv_head_num,
    head_dim,
    page_size,
    kv_dtype,
    sliding_window=None,
):
    data_factory = functools.partial(
        create_prefill_uniform_data,
        max_context_len,
        max_kv_cache_tokens,
        max_num_batched_tokens,
        q_head_num,
        kv_head_num,
        head_dim,
        page_size=page_size,
        dtype=kv_dtype,
    )
    return _run_attention_benchmark(
        data_factory,
        RpaCase.MIXED,
        q_head_num,
        kv_head_num,
        head_dim,
        page_size,
        max_num_batched_tokens,
        sliding_window=sliding_window,
    )


def benchmark_decode_backend(
    max_context_len,
    max_kv_cache_tokens,
    prefix_len,
    max_num_batched_tokens,
    q_head_num,
    kv_head_num,
    head_dim,
    page_size,
    kv_dtype,
    sliding_window=None,
):
    data_factory = functools.partial(
        create_decode_uniform_data,
        max_context_len,
        max_kv_cache_tokens,
        prefix_len,
        max_num_batched_tokens,
        q_head_num,
        kv_head_num,
        head_dim,
        page_size=page_size,
        dtype=kv_dtype,
    )
    return _run_attention_benchmark(
        data_factory,
        RpaCase.DECODE,
        q_head_num,
        kv_head_num,
        head_dim,
        page_size,
        max_num_batched_tokens,
        sliding_window=sliding_window,
    )


PAGE_SIZE_CONFIG = [256]
Q_KV_HEAD_NUM_CONFIG = [(12, 1), (24, 2), (48, 4), (96, 8)]
HEAD_DIM_CONFIG = [128, 256, 512]
MAX_KV_CACHE_TOKENS_CONFIG = [600000]
MAX_CONTEXT_LEN = 40960
MAX_NUM_BATCHED_TOKENS_CONFIG_FOR_PREFILL = [1024, 2048, 4096, 8192, 16384, 32768]
DECODE_PREFIX_LEN_CONFIG = [1024, 4096, 8192, 16384, 32768]
MAX_NUM_BATCHED_TOKENS_CONFIG_FOR_DECODE = [32, 64, 128]

KV_DTYPE = jnp.bfloat16
# KV_DTYPE = jnp.float8_e4m3fn

tpu_info = pltpu.get_tpu_info()


def _iter_head_configs():
    for q_head_num, kv_head_num in Q_KV_HEAD_NUM_CONFIG:
        for head_dim in HEAD_DIM_CONFIG:
            for page_size in PAGE_SIZE_CONFIG:
                for max_kv_cache_tokens in MAX_KV_CACHE_TOKENS_CONFIG:
                    yield (
                        q_head_num,
                        kv_head_num,
                        head_dim,
                        page_size,
                        max_kv_cache_tokens,
                    )


def prefill_benchmark():
    print("[PREFILL] BENCHMARK RESULTS SUMMARY")
    for (
        q_head_num,
        kv_head_num,
        head_dim,
        page_size,
        max_kv_cache_tokens,
    ) in _iter_head_configs():
        for max_num_batched_tokens in MAX_NUM_BATCHED_TOKENS_CONFIG_FOR_PREFILL:
            print(
                f"Config: q_head_num={q_head_num}, kv_head_num={kv_head_num}, head_dim={head_dim}, "
                f"max_num_batched_tokens={max_num_batched_tokens}, page_size={page_size}"
            )
            try:
                time_ms = benchmark_prefill_backend(
                    MAX_CONTEXT_LEN,
                    max_kv_cache_tokens,
                    max_num_batched_tokens,
                    q_head_num,
                    kv_head_num,
                    head_dim,
                    page_size,
                    KV_DTYPE,
                )
            except Exception as e:
                raise ValueError(f"run failed: {e=}")

            flops = 2 * max_num_batched_tokens * (max_num_batched_tokens + 512) * q_head_num * head_dim
            speed = flops / time_ms * 1000
            mfu = speed / tpu_info.bf16_ops_per_second
            print(f"cost: {time_ms:.4}ms, mfu: {mfu * 100:.1f}%")


def decode_benchmark():
    print("[DECODE] BENCHMARK RESULTS SUMMARY")
    for (
        q_head_num,
        kv_head_num,
        head_dim,
        page_size,
        max_kv_cache_tokens,
    ) in _iter_head_configs():
        for prefix_len in DECODE_PREFIX_LEN_CONFIG:
            for max_num_batched_tokens in MAX_NUM_BATCHED_TOKENS_CONFIG_FOR_DECODE:
                print(
                    f"Config: q_head_num={q_head_num}, kv_head_num={kv_head_num}, head_dim={head_dim}, "
                    f"prefix_len={prefix_len}, max_num_batched_tokens={max_num_batched_tokens}, "
                    f"page_size={page_size}"
                )
                try:
                    time_ms = benchmark_decode_backend(
                        MAX_CONTEXT_LEN,
                        max_kv_cache_tokens,
                        prefix_len,
                        max_num_batched_tokens,
                        q_head_num,
                        kv_head_num,
                        head_dim,
                        page_size,
                        KV_DTYPE,
                    )
                except Exception as e:
                    raise ValueError(f"run failed: {e=}")

                rw_bytes = max_num_batched_tokens * head_dim * ((prefix_len + 1) * 2 * kv_head_num + 2 * q_head_num) * 2
                throughput = rw_bytes / time_ms * 1000
                mbu = throughput / tpu_info.mem_bw_bytes_per_second
                print(f"cost: {time_ms:.4}ms, mbu: {mbu * 100:.1f}%")


if __name__ == "__main__":
    print("Run Ragged Paged Attention Full Benchmark...")
    prefill_benchmark()
    decode_benchmark()

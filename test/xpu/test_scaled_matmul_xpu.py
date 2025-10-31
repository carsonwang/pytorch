# Owner(s): ["module: intel"]

import unittest
from typing import Optional

import torch
from torch.nn.functional import ScalingType, SwizzleType
from torch.testing._internal.common_device_type import (
    E4M3_MAX_POS,
    e4m3_type,
    E5M2_MAX_POS,
    e5m2_type,
    instantiate_device_type_tests,
    onlyXPU,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase

from torch.testing._internal.common_quantized import (
    _f32_to_floatx_unpacked,
    _floatx_unpacked_to_f32,
    ceil_div, to_blocked,
    to_mxfp8,
    generate_jagged_offs,
)


f8_msg = "FP8 is not supported on this device"
f8_grouped_msg = "FP8 grouped is not supported on this device"
# avoid division by zero when calculating scale
EPS = 1e-12


def amax_to_scale(
    amax: torch.Tensor, float8_dtype: torch.dtype, orig_dtype: torch.dtype
):
    """Converts the amax value of a tensor to the fp8 scale.
    Args:
        amax: The amax value of the tensor.
        float8_dtype: the float8 dtype.
        orig_dtype: The original dtype of the tensor.
    """
    scale = torch.empty_like(amax, dtype=torch.float32)
    if float8_dtype == e4m3_type:
        res = E4M3_MAX_POS / torch.clamp(amax, min=EPS)
    elif float8_dtype == e5m2_type:
        res = E5M2_MAX_POS / torch.clamp(amax, min=EPS)
    else:
        raise ValueError(f"Unsupported float8_dtype: {float8_dtype}")

    # Ensure the scale is representable in float16,
    # this helps when amax is small. We are assuming that we don't need
    # to care about this for float32/bfloat16
    if orig_dtype is torch.float16:
        res = torch.clamp(res, max=torch.finfo(torch.float16).max)

    scale.copy_(res)
    return scale


def tensor_to_scale(x: torch.Tensor, float8_dtype: torch.dtype, dim=None):
    if dim is None:
        amax = torch.max(torch.abs(x))
    else:
        amax = torch.max(torch.abs(x), dim=dim, keepdim=True).values

    return amax_to_scale(amax, float8_dtype, x.dtype)


def tensor_to_scale_block(
    x: torch.Tensor,
    float8_dtype: torch.dtype,
    block_outer: int,
    block_inner: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.unflatten(1, (-1, block_inner)).unflatten(0, (-1, block_outer))
    amax = x.abs().amax(dim=[1, 3], keepdim=True).float()
    scale = torch.finfo(float8_dtype).max / amax
    x = x.mul(scale).to(float8_dtype)
    x = x.flatten(2, 3).flatten(0, 1)
    scale = scale.flatten(2, 3).flatten(0, 1)
    return x, scale


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y

def scaled_mm_wrap(
    a,
    b,
    scale_a,
    scale_b,
    scale_result=None,
    out_dtype=torch.bfloat16,
    use_fast_accum=False,
    bias=None,
):
    # Basically this don't need to be wrapped. Since there is a new API for scaled_mm_v2
    # We keep it here for future extension.
    # See https://github.com/pytorch/pytorch/pull/164142
    return torch._scaled_mm(
        a,
        b,
        scale_a,
        scale_b,
        scale_result=scale_result,
        out_dtype=out_dtype,
        bias=bias,
        use_fast_accum=use_fast_accum,
    )


def mm_float8_emulated(x, x_scale, y, y_scale, out_dtype) -> torch.Tensor:
    # naive implementation: dq -> op -> q
    x_fp32 = x.to(torch.float) / x_scale
    y_fp32 = y.to(torch.float) / y_scale
    out_fp32 = torch.mm(x_fp32, y_fp32)

    return out_fp32.to(out_dtype)

def mm_float8_emulated_block(x, x_scale, y, y_scale, out_dtype) -> torch.Tensor:
    x = x.unflatten(1, (x_scale.shape[1], -1)).unflatten(0, (x_scale.shape[0], -1))
    y = y.unflatten(1, (y_scale.shape[1], -1)).unflatten(0, (y_scale.shape[0], -1))
    x_fp32 = x.to(torch.float) / x_scale[:, None, :, None]
    y_fp32 = y.to(torch.float) / y_scale[:, None, :, None]
    x_fp32 = x_fp32.flatten(2, 3).flatten(0, 1)
    y_fp32 = y_fp32.flatten(2, 3).flatten(0, 1)
    out_fp32 = torch.mm(x_fp32, y_fp32)

    return out_fp32.to(out_dtype)

def addmm_float8_unwrapped(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    b_data: torch.Tensor,
    b_scale: torch.tensor,
    output_dtype: torch.dtype,
    output_scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    a_inverse_scale = a_scale.reciprocal()
    b_inverse_scale = b_scale.reciprocal()
    if output_dtype == torch.float32 and bias is not None:
        # Bias is not supported by _scaled_mm when output is fp32
        output = scaled_mm_wrap(
            a_data,
            b_data,
            scale_a=a_inverse_scale,
            scale_b=b_inverse_scale,
            scale_result=output_scale,
            out_dtype=output_dtype,
        )
        output += bias
        return output
    output = scaled_mm_wrap(
        a_data,
        b_data,
        bias=bias,
        scale_a=a_inverse_scale,
        scale_b=b_inverse_scale,
        scale_result=output_scale,
        out_dtype=output_dtype,
    )
    return output


def mm_float8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    output_dtype: torch.dtype,  # output dtype
    output_scale: Optional[torch.Tensor] = None,  # output scale, precomputed
) -> torch.Tensor:
    return addmm_float8_unwrapped(
        a, a_scale, b, b_scale, output_dtype, output_scale
    )


def to_fp8_saturated(x: torch.Tensor, fp8_dtype: torch.dtype):
    if fp8_dtype == e4m3_type:
        x = x.clamp(min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)
    elif fp8_dtype == e5m2_type:
        x = x.clamp(min=-1 * E5M2_MAX_POS, max=E5M2_MAX_POS)
    else:
        raise ValueError(f"to_fp8_saturated(): Unsupported fp8_dtype: {fp8_dtype}")

    return x.to(fp8_dtype)


def compute_error(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the error between two tensors in dB.

    For more details see:
        https://en.wikipedia.org/wiki/Signal-to-noise_ratio

    Args:
        x: The original tensor.
        y: The tensor to compare to the original tensor.
    """
    Ps = torch.norm(x)
    Pn = torch.norm(x - y)
    return 20 * torch.log10(Ps / Pn)


# largest power of 2 representable in `torch.float8_e4m3fn`
F8E4M3_LARGEST_POW2 = 8
# largest power of 2 representable in `torch.float4_e2m1fn_x2`
FP4E2M1FN_LARGEST_POW2 = 2.0
# max value of `torch.float8_e4m3fn` (448)
F8E4M3_MAX_VAL = torch.finfo(torch.float8_e4m3fn).max
# exponent bias of `torch.float8_e8m0fnu`
F8E8M0_EXP_BIAS = 127
# exponent and mantissa bits of `torch.float4_e2m1fn_x2`
FP4_EBITS, FP4_MBITS = 2, 1
FP4_MAX_VAL = 6.0

def data_to_mx_scale(x, block_size, recipe):
    # simple implementation of https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    # section 6.3, not all edge cases (such as NaN) are handled/tested
    if recipe == "mxfp8":
        largest_pow2 = F8E4M3_LARGEST_POW2
    elif recipe == "mxfp4":
        largest_pow2 = FP4E2M1FN_LARGEST_POW2
    else:
        raise ValueError(f"data_to_mx_scale(): Unsupported mx recipe: {recipe}")
    orig_shape = x.shape
    x = x.reshape(-1, block_size)
    max_abs = torch.amax(torch.abs(x), 1)
    largest_p2_lt_max_abs = torch.floor(torch.log2(max_abs))
    scale_e8m0_unbiased = largest_p2_lt_max_abs - largest_pow2
    scale_e8m0_unbiased = torch.clamp(scale_e8m0_unbiased, -1 * F8E8M0_EXP_BIAS, F8E8M0_EXP_BIAS)
    scale_e8m0_biased = scale_e8m0_unbiased + F8E8M0_EXP_BIAS
    scale_e8m0_biased = scale_e8m0_biased.to(torch.uint8)
    scale_e8m0_biased = scale_e8m0_biased.view(torch.float8_e8m0fnu)
    return scale_e8m0_biased.reshape(orig_shape[0], -1)


def down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


def pack_uint4(uint8_data) -> torch.Tensor:
    # converting to uint8 for operations
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[1::2] << 4 | uint8_data[::2]).view(down_size(shape))

def _bfloat16_to_float4_e2m1fn_x2(x):
    assert x.dtype == torch.bfloat16
    x = _f32_to_floatx_unpacked(x.float(), FP4_EBITS, FP4_MBITS)
    x = pack_uint4(x)
    x = x.view(torch.float4_e2m1fn_x2)
    return x

# Encode a power-of-two scale s = 2^k into E8M0 byte
def e8m0_pow2_byte(k: int) -> int:  # clamp to valid range [-127,127]
    k = max(-127, min(127, k))
    return 127 + k  # E8M0 stores biased exponent

class TestFP8Matmul(TestCase):
    def _test_tautological_mm(
        self,
        device: str = "xpu",
        x_dtype: torch.dtype = e4m3_type,
        y_dtype: torch.dtype = e4m3_type,
        out_dtype: Optional[torch.dtype] = None,
        size: int = 16,
    ) -> None:
        x_fp8 = torch.rand(size, size, device=device).to(x_dtype)
        y_fp8 = torch.eye(size, device=device, dtype=y_dtype).t()
        out_fp32 = torch.mm(x_fp8.to(torch.float), y_fp8.to(torch.float))
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)
        out_fp8 = scaled_mm_wrap(x_fp8, y_fp8, scale_a, scale_b, out_dtype=out_dtype)
        if out_dtype is not None:
            self.assertEqual(out_dtype, out_fp8.dtype)
        self.assertEqual(out_fp32, out_fp8.to(torch.float))

    def test_float8_basics(self, device="xpu") -> None:
        self._test_tautological_mm(device, e4m3_type, e4m3_type, size=16)

        self._test_tautological_mm(device, e5m2_type, e5m2_type)

        self._test_tautological_mm(device, e4m3_type, e5m2_type, size=32)
        self._test_tautological_mm(device, e5m2_type, e4m3_type, size=48)

        self._test_tautological_mm(device, size=64, out_dtype=torch.float16)
        self._test_tautological_mm(device, size=96, out_dtype=torch.float32)
        self._test_tautological_mm(device, size=80, out_dtype=torch.bfloat16)

        with self.assertRaises(AssertionError):
            self._test_tautological_mm(device, out_dtype=e5m2_type)

    def test_float8_scale(self, device="xpu") -> None:
        size = (16, 16)
        x = torch.full(size, 0.5, device=device, dtype=e4m3_type)
        y_type = e5m2_type
        y = torch.full(size, 0.5, device=device, dtype=y_type).t()
        scale_one = torch.tensor(1.0, device=device)
        scale_a = torch.tensor(1.5, device=device)
        scale_b = torch.tensor(0.66, device=device)
        out_fp8 = scaled_mm_wrap(x, y, scale_a=scale_one, scale_b=scale_one)
        self.assertEqual(out_fp8.to(torch.float), torch.full(size, 4.0, device=device))
        out_fp8_s = scaled_mm_wrap(x, y, scale_a=scale_a, scale_b=scale_b)
        self.assertEqual(out_fp8, out_fp8_s)

    @parametrize("base_dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_scaled_mm_vs_emulated(self, base_dtype):
        torch.manual_seed(42)
        input_dtype = e4m3_type
        output_dtype = base_dtype
        compare_type = torch.float32

        x = torch.randn(16, 16, device="xpu", dtype=base_dtype)
        y = torch.randn(32, 16, device="xpu", dtype=base_dtype).t()

        x_scale = tensor_to_scale(x, input_dtype).float()
        y_scale = tensor_to_scale(y, input_dtype).float()

        x_fp8 = to_fp8_saturated(x * x_scale, input_dtype)
        y_fp8 = to_fp8_saturated(y * y_scale, input_dtype)

        # Calculate actual F8 mm
        out_scaled_mm = scaled_mm_wrap(
            x_fp8,
            y_fp8,
            scale_a=x_scale.reciprocal(),
            scale_b=y_scale.reciprocal(),
            out_dtype=output_dtype,
        )

        # Calculate emulated F8 mm
        out_emulated = mm_float8_emulated(x_fp8, x_scale, y_fp8, y_scale, output_dtype)

        if output_dtype != base_dtype:
            out_scaled_mm = out_scaled_mm.to(compare_type)
            out_scaled_mm = out_scaled_mm / tensor_to_scale(out_scaled_mm, input_dtype)

            out_emulated = out_emulated.to(compare_type)
            out_emulated = out_emulated / tensor_to_scale(out_emulated, input_dtype)

        if base_dtype in {torch.bfloat16, torch.float16}:
            atol, rtol = 7e-2, 7e-2
        else:
            atol, rtol = 3e-3, 3e-3

        torch.testing.assert_close(out_scaled_mm, out_emulated, atol=atol, rtol=rtol)

    @parametrize("base_dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_scaled_mm_change_stride(self, base_dtype):
        torch.manual_seed(42)
        input_dtype = e4m3_type
        output_dtype = base_dtype
        compare_type = torch.float32

        x = torch.empty_strided((16, 16), (16, 1), device="xpu", dtype=base_dtype)
        y = torch.empty_strided((16, 32), (1, 64), device="xpu", dtype=base_dtype)

        x.normal_()
        y.normal_()

        x_scale = tensor_to_scale(x, input_dtype).float()
        y_scale = tensor_to_scale(y, input_dtype).float()

        x_fp8 = to_fp8_saturated(x * x_scale, input_dtype)
        y_fp8 = to_fp8_saturated(y * y_scale, input_dtype)

        # Calculate actual F8 mm
        out_scaled_mm = scaled_mm_wrap(
            x_fp8,
            y_fp8,
            scale_a=x_scale.reciprocal(),
            scale_b=y_scale.reciprocal(),
            out_dtype=output_dtype,
        )

        # Calculate emulated F8 mm
        out_emulated = mm_float8_emulated(x_fp8, x_scale, y_fp8, y_scale, output_dtype)

        if output_dtype != base_dtype:
            out_scaled_mm = out_scaled_mm.to(compare_type)
            out_scaled_mm = out_scaled_mm / tensor_to_scale(out_scaled_mm, input_dtype)

            out_emulated = out_emulated.to(compare_type)
            out_emulated = out_emulated / tensor_to_scale(out_emulated, input_dtype)

        if base_dtype in {torch.bfloat16, torch.float16}:
            atol, rtol = 7e-2, 7e-2
        else:
            atol, rtol = 3e-3, 3e-3

        torch.testing.assert_close(out_scaled_mm, out_emulated, atol=atol, rtol=rtol)

    @onlyXPU
    def test_float8_bias(self, device) -> None:
        (k, l, m) = (16, 48, 32)
        x = torch.ones((k, l), device=device).to(e4m3_type)
        y = torch.full((m, l), 0.25, device=device, dtype=e4m3_type).t()
        bias = torch.full((m,), 4.0, device=device, dtype=torch.bfloat16)
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)
        out_fp8 = scaled_mm_wrap(x, y, scale_a=scale_a, scale_b=scale_b)
        outb_fp8 = scaled_mm_wrap(x, y, scale_a=scale_a, scale_b=scale_b, bias=bias)

        out_fp32 = out_fp8.to(torch.float32)
        outb_fp32 = outb_fp8.to(torch.float32)
        difference = torch.abs(out_fp32 - outb_fp32)
        self.assertEqual(
            difference, torch.tensor(4.0, device=device).expand_as(out_fp32)
        )

    @onlyXPU
    @parametrize("bias", [True, False])
    def test_non_divisible_leading_dim(self, device, bias: bool) -> None:
        x = torch.rand((17, 16), device=device).to(e4m3_type)
        y = torch.rand((16, 16), device=device).to(e4m3_type).t()
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)
        input_bias = None
        if bias:
            input_bias = torch.rand((16,), device=device).to(torch.bfloat16)
        _ = scaled_mm_wrap(x, y, scale_a, scale_b, bias=input_bias)

    @onlyXPU
    def test_float8_bias_relu_edgecase(self, device) -> None:
        (k, l, m) = (16, 48, 32)
        x = torch.full((k, l), 0.0, device=device).to(e4m3_type)
        y = torch.full((m, l), 1.0, device=device, dtype=e4m3_type).t()
        bias = torch.full((m,), -3.0, device=device, dtype=torch.bfloat16)
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)
        outb_fp8 = scaled_mm_wrap(x, y, scale_a, scale_b, bias=bias)
        outb_fp32 = outb_fp8.to(torch.float32)
        self.assertEqual(
            outb_fp32, torch.tensor(-3.0, device=device).expand_as(outb_fp32)
        )

    @onlyXPU
    def test_float32_output_errors_with_bias(self, device) -> None:
        (k, l, m) = (16, 48, 32)
        x = torch.rand((k, l), device=device).to(e4m3_type)
        y = torch.full((m, l), 0.25, device=device, dtype=e4m3_type).t()
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)
        bias = torch.full((m,), 4.0, device=device, dtype=torch.bfloat16)
        self.assertRaisesRegex(
            RuntimeError,
            "Bias is not supported when out_dtype is set to Float32",
            lambda: scaled_mm_wrap(
                x, y, scale_a, scale_b, bias=bias, out_dtype=torch.float32
            ),
        )

    @onlyXPU
    @parametrize("output_dtype", [torch.bfloat16, torch.float32])
    @parametrize("lhs_block,rhs_block", [(1, 1)])
    @parametrize("M,N,K", [(128, 512, 256)])  # TODO: @parametrize("M,N,K", [(256, 768, 512)])
    def test_scaled_mm_vs_emulated_block_wise(self, output_dtype, lhs_block, rhs_block, M, N, K):
        torch.manual_seed(42)

        x = torch.randn(M, K, device="xpu", dtype=output_dtype).pow(3)
        y = torch.randn(N, K, device="xpu", dtype=output_dtype).pow(3)

        x_fp8, x_scales = tensor_to_scale_block(x, e4m3_type, lhs_block, 128)
        y_fp8, y_scales = tensor_to_scale_block(y, e4m3_type, rhs_block, 128)

        # 1x128 blocks need scales to be outer-dim-major
        if lhs_block == 1:
            x_scales = x_scales.t().contiguous().t()
        if rhs_block == 1:
            y_scales = y_scales.t().contiguous().t()

        # Calculate actual F8 mm
        out_scaled_mm = mm_float8(
            x_fp8, y_fp8.t(), a_scale=x_scales, b_scale=y_scales.t(), output_dtype=output_dtype
        )

        # Calculate emulated F8 mm
        out_emulated = mm_float8_emulated_block(
            x_fp8, x_scales, y_fp8.t(), y_scales.t(), output_dtype
        )

        cosine_sim = torch.nn.functional.cosine_similarity(
            out_scaled_mm.flatten().float(), out_emulated.flatten().float(), dim=0
        )
        self.assertGreaterEqual(float(cosine_sim), 0.999)

        if output_dtype in {torch.bfloat16, torch.float16}:
            atol, rtol = 6e-1, 7e-2
        else:
            atol, rtol = 7e-1, 2e-3

        self.assertEqual(out_scaled_mm, out_emulated, atol=atol, rtol=rtol)

        # One last check against the full-precision reference, to ensure we
        # didn't mess up the scaling itself and made the test trivial.
        cosine_sim = torch.nn.functional.cosine_similarity(
            out_scaled_mm.flatten().float(), (x @ y.t()).flatten().float(), dim=0
        )
        self.assertGreaterEqual(float(cosine_sim), 0.999)

    # TODO: Some of the tests need to rely on MXFP8/MXFP4 ops. MXFP8 is tracked in https://github.com/intel/torch-xpu-ops/issues/2207
    @onlyXPU
    @parametrize("test_case_name", [
        "a_eye_b_eye",
        "a_ones_b_ones",
        "a_ones_modified_b_ones",
        "a_ones_b_ones_modified",
        "a_scale_modified_b_ones",
        "a_ones_b_scale_modified",
        "data_random_scales_one",
        # "data_random_scales_from_data",
    ])
    @parametrize("fast_accum", [False])
    @parametrize("mkn", [
        # Nice shapes
        (128, 128, 128),
        (256, 256, 256),
        (128, 256, 512),
        (256, 512, 128),
        (512, 128, 256),

        # Non block multiples
        (65, 96, 112),
        (197, 224, 272),
        # K not multiple of 32 (skipped for fp4)
        (197, 240, 272),

        # Very unbalanced
        (1023, 64, 48),
        (31, 1024, 64),
        (45, 96, 1024),

        # Mixed large and small
        (2, 1024, 128),
        (127, 96, 1024),
        (1025, 128, 96)
    ], name_fn=lambda mkn: f"{mkn[0]}_{mkn[1]}_{mkn[2]}")
    @parametrize("recipe", ["mxfp8"]) # // TODO: add , nvfp4+
    def test_blockwise_mxfp8_nvfp4_mxfp4_numerics(self, test_case_name, fast_accum, mkn, recipe) -> None:
        device = "xpu"
        M, K, N = mkn
        if recipe == "mxfp4" and K % 32 != 0:
            raise unittest.SkipTest("K must be divisible by 32 for mxfp4 gemm, skipping")

        fp4_scaling_dtype = torch.float8_e8m0fnu
        BLOCK_SIZE = 16 if recipe == "nvfp4" else 32
        require_exact_match = True
        approx_match_sqnr_target = 22.0

        if test_case_name == "a_eye_b_eye":
            if not ((M == K) and (M == N)):
                raise unittest.SkipTest("this test is only defined for M == K == N, skipping")
            A_ref = torch.eye(M, device=device, dtype=torch.bfloat16)
            B_ref = torch.eye(M, device=device, dtype=torch.bfloat16)

            if recipe == "mxfp8":
                A = A_ref.to(torch.float8_e4m3fn)
                B = B_ref.to(torch.float8_e4m3fn)
                A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu)
                B_scale = torch.full((ceil_div(K, BLOCK_SIZE), N), 1.0, device=device, dtype=torch.float8_e8m0fnu)
            else:  # nvfp4 # mxfp4
                A = _bfloat16_to_float4_e2m1fn_x2(A_ref)
                B = _bfloat16_to_float4_e2m1fn_x2(B_ref)
                A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=fp4_scaling_dtype)
                B_scale = torch.full((ceil_div(K, BLOCK_SIZE), N), 1.0, device=device, dtype=fp4_scaling_dtype)

        elif test_case_name == "a_ones_b_ones":
            A_ref = torch.ones(M, K, device=device, dtype=torch.bfloat16)
            B_ref = torch.ones(N, K, device=device, dtype=torch.bfloat16)

            if recipe == "mxfp8":
                A = A_ref.to(torch.float8_e4m3fn)
                B = B_ref.to(torch.float8_e4m3fn)
                A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu)
                B_scale = torch.full((ceil_div(K, BLOCK_SIZE), N), 1.0, device=device, dtype=torch.float8_e8m0fnu)
            else:  # nvfp4 # mxfp4
                A = _bfloat16_to_float4_e2m1fn_x2(A_ref)
                B = _bfloat16_to_float4_e2m1fn_x2(B_ref)
                A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=fp4_scaling_dtype)
                B_scale = torch.full((ceil_div(K, BLOCK_SIZE), N), 1.0, device=device, dtype=fp4_scaling_dtype)

        elif test_case_name == "a_ones_modified_b_ones":
            A_ref = torch.ones(M, K, device=device, dtype=torch.bfloat16)
            B_ref = torch.ones(N, K, device=device, dtype=torch.bfloat16)
            A_ref[1][0:BLOCK_SIZE] = 2

            if recipe == "mxfp8":
                A = A_ref.to(torch.float8_e4m3fn)
                B = B_ref.to(torch.float8_e4m3fn)
                A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu)
                B_scale = torch.full((ceil_div(K, BLOCK_SIZE), N), 1.0, device=device, dtype=torch.float8_e8m0fnu)
            else:  # nvfp4 # mxfp4
                A = _bfloat16_to_float4_e2m1fn_x2(A_ref)
                B = _bfloat16_to_float4_e2m1fn_x2(B_ref)
                A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=fp4_scaling_dtype)
                B_scale = torch.full((ceil_div(K, BLOCK_SIZE), N), 1.0, device=device, dtype=fp4_scaling_dtype)

        elif test_case_name == "a_ones_b_ones_modified":
            A_ref = torch.ones(M, K, device=device, dtype=torch.bfloat16)
            B_ref = torch.ones(N, K, device=device, dtype=torch.bfloat16)
            B_ref[1][0:BLOCK_SIZE] = 2


            if recipe == "mxfp8":
                A = A_ref.to(torch.float8_e4m3fn)
                B = B_ref.to(torch.float8_e4m3fn)
                A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu)
                B_scale = torch.full((ceil_div(K, BLOCK_SIZE), N), 1.0, device=device, dtype=torch.float8_e8m0fnu)
            else:  # nvfp4 # mxfp4
                A = _bfloat16_to_float4_e2m1fn_x2(A_ref)
                B = _bfloat16_to_float4_e2m1fn_x2(B_ref)
                A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=fp4_scaling_dtype)
                B_scale = torch.full((ceil_div(K, BLOCK_SIZE), N), 1.0, device=device, dtype=fp4_scaling_dtype)

        elif test_case_name == "a_scale_modified_b_ones":
            A_ref = torch.ones(M, K, device=device, dtype=torch.bfloat16)
            B_ref = torch.ones(N, K, device=device, dtype=torch.bfloat16)

            if recipe == "mxfp8":
                A = A_ref.to(torch.float8_e4m3fn)
                B = B_ref.to(torch.float8_e4m3fn)
                # A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu)
                # B_scale = torch.full((N, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu)
                # A_ref[1][0:BLOCK_SIZE] = 4
                # A[1][0:BLOCK_SIZE] = 2
                # A_scale[1][0] = 2

                # Workaround before Float8_e8m0fnu copy is supported.
                A_scale_u8 = torch.full((M, ceil_div(K, BLOCK_SIZE)), e8m0_pow2_byte(0), device=device, dtype=torch.uint8)
                B_scale_u8 = torch.full((ceil_div(K, BLOCK_SIZE), N), e8m0_pow2_byte(0), device=device, dtype=torch.uint8)
                A_ref[1][0:BLOCK_SIZE] = 4
                A[1][0:BLOCK_SIZE] = 2
                A_scale_u8[1, 0] = e8m0_pow2_byte(1)
                A_scale = A_scale_u8.view(torch.float8_e8m0fnu)
                B_scale = B_scale_u8.view(torch.float8_e8m0fnu)

            else:  # nvfp4 # mxfp4
                A = _bfloat16_to_float4_e2m1fn_x2(A_ref)
                B = _bfloat16_to_float4_e2m1fn_x2(B_ref)
                A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=fp4_scaling_dtype)
                B_scale = torch.full((N, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=fp4_scaling_dtype)
                A_ref[1][0:BLOCK_SIZE] = 4
                A.view(torch.uint8)[1][0:(BLOCK_SIZE // 2)] = 0b01000100
                A_scale[1][0] = 2

        elif test_case_name == "a_ones_b_scale_modified":
            A_ref = torch.ones(M, K, device=device, dtype=torch.bfloat16)
            B_ref = torch.ones(N, K, device=device, dtype=torch.bfloat16)

            if recipe == "mxfp8":
                A = A_ref.to(torch.float8_e4m3fn)
                B = B_ref.to(torch.float8_e4m3fn)
                # A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu)
                # B_scale = torch.full((N, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu)
                # B_ref[1][0:BLOCK_SIZE] = 4
                # B[1][0:BLOCK_SIZE] = 2
                # B_scale[1][0] = 2

                # Workaround before Float8_e8m0fnu copy is supported.
                A_scale_u8 = torch.full((M, ceil_div(K, BLOCK_SIZE)), e8m0_pow2_byte(0), device=device, dtype=torch.uint8)
                B_scale_u8 = torch.full((ceil_div(K, BLOCK_SIZE), N), e8m0_pow2_byte(0), device=device, dtype=torch.uint8)
                B_ref[1][0:BLOCK_SIZE] = 4
                B[1][0:BLOCK_SIZE] = 2
                B_scale_u8[0][1] = e8m0_pow2_byte(1)
                A_scale = A_scale_u8.view(torch.float8_e8m0fnu)
                B_scale = B_scale_u8.view(torch.float8_e8m0fnu)
            else:  # nvfp4 # mxfp4
                A = _bfloat16_to_float4_e2m1fn_x2(A_ref)
                B = _bfloat16_to_float4_e2m1fn_x2(B_ref)
                A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=fp4_scaling_dtype)
                B_scale = torch.full((N, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=fp4_scaling_dtype)
                B_ref[1][0:BLOCK_SIZE] = 4
                B.view(torch.uint8)[1][0:(BLOCK_SIZE // 2)] = 0b01000100
                B_scale[1][0] = 2

        elif test_case_name == "data_random_scales_one":
            require_exact_match = False

            if recipe == "mxfp8":
                # scales all-ones, element data random while being exactly representable in float8_e4m3fn
                # generate integers in [0, 255] and interpret as float8_e4m3fn
                A_ref = torch.randint(0, 255, (M, K), device=device, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.bfloat16)
                B_ref = torch.randint(0, 255, (N, K), device=device, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.bfloat16)
                # modification: don't allow NaN values
                A_ref[torch.isnan(A_ref)] = 0
                B_ref[torch.isnan(B_ref)] = 0
                A = A_ref.to(torch.float8_e4m3fn)
                B = B_ref.to(torch.float8_e4m3fn)
                A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu)
                B_scale = torch.full((ceil_div(K, BLOCK_SIZE), N), 1.0, device=device, dtype=torch.float8_e8m0fnu)
            else:  # nvfp4 # mxfp4
                # scales all-ones, element data random while being exactly representable in float4_e2m1fn_x2
                # generate integers in [0, 16] and cast to bfloat16
                A_ref = _floatx_unpacked_to_f32(
                    torch.randint(0, 16, (M, K), device=device, dtype=torch.uint8),
                    FP4_EBITS,
                    FP4_MBITS
                ).bfloat16()
                B_ref = _floatx_unpacked_to_f32(
                    torch.randint(0, 16, (N, K), device=device, dtype=torch.uint8),
                    FP4_EBITS,
                    FP4_MBITS
                ).bfloat16()
                A = _bfloat16_to_float4_e2m1fn_x2(A_ref)
                B = _bfloat16_to_float4_e2m1fn_x2(B_ref)
                A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=fp4_scaling_dtype)
                B_scale = torch.full((N, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=fp4_scaling_dtype)

        elif test_case_name == "data_random_scales_from_data":
            if not K % BLOCK_SIZE == 0:
                raise unittest.SkipTest(f"this test is only defined for K a multiple of {BLOCK_SIZE}, skipping")
            require_exact_match = False
            # random data, scales from data
            A_ref = torch.randn((M, K), device=device, dtype=torch.bfloat16) * 1000
            B_ref = torch.randn((N, K), device=device, dtype=torch.bfloat16) * 1000

            if recipe == "mxfp8":
                # Calculate scales based on the inputs
                A_scale = data_to_mx_scale(A_ref, BLOCK_SIZE, recipe)
                B_scale = data_to_mx_scale(B_ref, BLOCK_SIZE, recipe)
                max_val = F8E4M3_MAX_VAL
                min_val = -1 * max_val
                A = (A_ref.reshape(-1, BLOCK_SIZE) / A_scale.reshape(M * ceil_div(K, BLOCK_SIZE), 1).float()).reshape(M, K)
                A = A.clamp(min=min_val, max=max_val).to(torch.float8_e4m3fn)
                B = (B_ref.reshape(-1, BLOCK_SIZE) / B_scale.reshape(N * ceil_div(K, BLOCK_SIZE), 1).float()).reshape(N, K)
                B = B.clamp(min=min_val, max=max_val).to(torch.float8_e4m3fn)
            else:  # nvfp4 # mxfp4
                if recipe == "mxfp4":
                    A_scale = data_to_mx_scale(A_ref, BLOCK_SIZE, recipe)
                    B_scale = data_to_mx_scale(B_ref, BLOCK_SIZE, recipe)
                # else:
                #     A_scale = data_to_nvfp4_scale(A_ref, BLOCK_SIZE)
                #     B_scale = data_to_nvfp4_scale(B_ref, BLOCK_SIZE)
                max_val = FP4_MAX_VAL
                min_val = -1 * max_val

                A = (A_ref.reshape(-1, BLOCK_SIZE) / A_scale.reshape(M * ceil_div(K, BLOCK_SIZE), 1).bfloat16()).reshape(M, K)
                A = A.clamp(min=min_val, max=max_val)
                A = _bfloat16_to_float4_e2m1fn_x2(A)
                B = (B_ref.reshape(-1, BLOCK_SIZE) / B_scale.reshape(N * ceil_div(K, BLOCK_SIZE), 1).bfloat16()).reshape(N, K)
                B = B.clamp(min=min_val, max=max_val)
                B = _bfloat16_to_float4_e2m1fn_x2(B)

                approx_match_sqnr_target = 15 if torch.version.hip else 15.8

        C_ref = A_ref @ B_ref.t()

        C = scaled_mm_wrap(
            A,
            B.t(),
            A_scale,
            B_scale,
            out_dtype=torch.bfloat16,
            use_fast_accum=fast_accum,
        )

        if require_exact_match:
            torch.testing.assert_close(C, C_ref, atol=0, rtol=0)
        else:
            sqnr = compute_error(C_ref, C)
            assert sqnr.item() > approx_match_sqnr_target


instantiate_device_type_tests(TestFP8Matmul, globals(), only_for="xpu", allow_xpu=True)

if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()

#!/usr/bin/env bash
# Row grammar regression: existing rows must emit exactly what they did before the _a<N> / ssmax additions.
#   bash tests/test_rows.sh
source "$(dirname "$0")/../osc/rows.sh"
fail=0
check() { local got; got=$(row_args "$1"); [ "$got" == "$2" ] && echo "OK  $1" || { echo "BAD $1: '$got' != '$2'"; fail=1; }; }
check softmax_xs                 "--attn_type=softmax --nape_slope_type=exp"
check asentmax_zi_xs             "--attn_type=entmax --entmax_alpha=1.5 --attn_scale=as --attn_scale_zero_init=0.05 --nape_slope_type=exp"
check wstieltjes_q3_d2_xs        "--attn_type=stieltjes --stieltjes_q=3.0 --stieltjes_window=2.0 --nape_slope_type=exp"
check aswstieltjes_q2_d4_zi      "--attn_type=stieltjes --stieltjes_q=2.0 --stieltjes_window=4.0 --attn_scale=as --attn_scale_zero_init=0.05"
check stieltjes_q4               "--attn_type=stieltjes --stieltjes_q=4.0"
check ssmax_xs                   "--attn_type=softmax --attn_scale=ssmax --nape_slope_type=exp"
check softmax_xs_a2              "--attn_type=softmax --nape_slope_type=exp_a2"
check wstieltjes_q3_d2_xs_a4     "--attn_type=stieltjes --stieltjes_q=3.0 --stieltjes_window=2.0 --nape_slope_type=exp_a4"
check asentmax_zi_xs_a0          "--attn_type=entmax --entmax_alpha=1.5 --attn_scale=as --attn_scale_zero_init=0.05 --nape_slope_type=exp_a0"
check softmax_a2                 "--attn_type=softmax --nape_slope_type=linear_a2"
exit $fail

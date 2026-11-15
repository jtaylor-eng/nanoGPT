#!/usr/bin/env bash
# Row grammar regression: existing rows must emit exactly what they did before the ssmax / adatemp additions.
#   bash tests/test_rows.sh
source "$(dirname "$0")/../osc/rows.sh"
fail=0
check() { local got; got=$(row_args "$1"); [ "$got" == "$2" ] && echo "OK  $1" || { echo "BAD $1: '$got' != '$2'"; fail=1; }; }
check softmax_xs                 "--attn_type=softmax --nape_slope_type=exp"
check asentmax_zi_xs             "--attn_type=entmax --entmax_alpha=1.5 --attn_scale=as --attn_scale_zero_init=0.05 --nape_slope_type=exp"
check wstieltjes_q3_d2_xs        "--attn_type=stieltjes --stieltjes_q=3.0 --stieltjes_window=2.0 --nape_slope_type=exp"
check wstieltjes_q3_d1.5_xs      "--attn_type=stieltjes --stieltjes_q=3.0 --stieltjes_window=1.5 --nape_slope_type=exp"
check aswstieltjes_q3_d2_zi_xs   "--attn_type=stieltjes --stieltjes_q=3.0 --stieltjes_window=2.0 --attn_scale=as --attn_scale_zero_init=0.05 --nape_slope_type=exp"
check stieltjes_q4               "--attn_type=stieltjes --stieltjes_q=4.0"
check ssmax_xs                   "--attn_type=softmax --attn_scale=ssmax --nape_slope_type=exp"
check adatemp_xs                 "--attn_type=softmax --attn_scale=adatemp --nape_slope_type=exp"
exit $fail

# row name -> train.py attention overrides (NAPE everywhere via config/lm_stieltjes.py)
row_args() {
  case $1 in
    softmax)               echo "--attn_type=softmax" ;;
    softmax_as_zi)         echo "--attn_type=softmax --attn_scale=as --attn_scale_zero_init=0.05" ;;
    entmax)                echo "--attn_type=entmax --entmax_alpha=1.5" ;;
    asentmax_zi)           echo "--attn_type=entmax --entmax_alpha=1.5 --attn_scale=as --attn_scale_zero_init=0.05" ;;
    stieltjes_q4)          echo "--attn_type=stieltjes --stieltjes_q=4.0" ;;
    wstieltjes_q4_d2)      echo "--attn_type=stieltjes --stieltjes_q=4.0 --stieltjes_window=2.0" ;;
    aswstieltjes_q2_d4_zi) echo "--attn_type=stieltjes --stieltjes_q=2.0 --stieltjes_window=4.0 --attn_scale=as --attn_scale_zero_init=0.05" ;;
    *) echo "unknown row $1" >&2; return 1 ;;
  esac
}
is_stieltjes() { [[ $1 == *stieltjes* ]]; }

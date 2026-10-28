# row name -> train.py attention overrides (NAPE everywhere via config/lm_stieltjes.py)
# Grammar:  <base>[_zi][_xs]
#   base: softmax | softmax_as | entmax | asentmax | [as][w]stieltjes[_q<Q>][_d<D>]
#   as  = adaptive scale (attn_scale=as);  _zi = zero-init the scale (eps 0.05; all AS rows so far use it)
#   w   = windowed Stieltjes; _q<Q> default 4; _d<D> default 2 for windowed, dense otherwise
#   _xs = exponential (original ALiBi) NAPE slopes instead of the synthetic-task 1/h slopes
row_args() {
  local name=$1 out="" xs="" zi=""
  [[ $name == *_xs ]] && { xs=" --nape_slope_type=exp"; name=${name%_xs}; }
  [[ $name == *_zi ]] && { zi=" --attn_scale_zero_init=0.05"; name=${name%_zi}; }
  case $name in
    softmax)     out="--attn_type=softmax" ;;
    softmax_as)  out="--attn_type=softmax --attn_scale=as" ;;
    entmax)      out="--attn_type=entmax --entmax_alpha=1.5" ;;
    asentmax)    out="--attn_type=entmax --entmax_alpha=1.5 --attn_scale=as" ;;
    *)
      if [[ $name =~ ^(as)?(w)?stieltjes(_q([0-9.]+))?(_d([0-9.]+))?$ ]]; then
        local as=${BASH_REMATCH[1]} w=${BASH_REMATCH[2]} q=${BASH_REMATCH[4]:-4} d=${BASH_REMATCH[6]}
        # configurator.py asserts type(value) == type(default) (float): always emit a decimal point
        [[ $q == *.* ]] || q="$q.0"; [ -n "$d" ] && { [[ $d == *.* ]] || d="$d.0"; }
        out="--attn_type=stieltjes --stieltjes_q=$q"
        if [ -n "$w" ]; then out="$out --stieltjes_window=${d:-2.0}"; elif [ -n "$d" ]; then echo "dense row with _d" >&2; return 1; fi
        [ -n "$as" ] && out="$out --attn_scale=as"
      else
        echo "unknown row $1" >&2; return 1
      fi ;;
  esac
  echo "$out$zi$xs"
}
is_stieltjes() { [[ $1 == *stieltjes* ]]; }

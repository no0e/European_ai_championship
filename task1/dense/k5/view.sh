#!/bin/bash
# view.sh <dossier source contenant f0..f4> <méthode> : affiche les 5 dossiers d une vue à une seule méthode (mix/v_<nom>_<méthode>/f<k>)
src=$1; m=$2; n=$(echo $src | tr "/." "__"); out=""
for f in 0 1 2 3 4; do d=mix/v_${n}_$m/f$f; mkdir -p $d
  for s in train test; do [ -e $src/f$f/${m}_$s.npz ] && ln -sf $(readlink -f $src/f$f/${m}_$s.npz) $d/${m}_$s.npz; done; out="$out $d"; done
echo $out

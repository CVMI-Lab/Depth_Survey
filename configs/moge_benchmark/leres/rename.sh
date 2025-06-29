for f in geowizard-align_*.yaml; do
  mv -- "$f" "leres-align_${f#geowizard-align_}"
done

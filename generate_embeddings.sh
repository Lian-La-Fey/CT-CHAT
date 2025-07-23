find ~/gcs/ct_rate/dataset/valid -type f -name '*.nii.gz' | while read -r nii; do
  echo "Encoding $nii …"
  python llava/serve/encode_script.py \
    --path "$nii" \
    --slope 1 --intercept 0 \
    --xy_spacing 1 --z_spacing 1
done
count=0
for file in /nfs/usrhome/khmuhammad/Echonet/samples/lidm_dynamic_with_ddpm/privacy_compliant_images/*; do
    if [ $count -lt 50000 ]; then
        cp "$file" /nfs/usrhome/khmuhammad/Echonet/samples/lidm_dynamic_with_ddpm/privacy_compliant_images_50k/
        ((count++))
    else
        break
    fi
done
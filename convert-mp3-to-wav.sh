for file in labeler-vue/public/library/*.mp3; do
  ffmpeg -i "$file" -ar 16000 -ac 1 "labeler-vue/public/library-wav/$(basename "${file%.*}").wav"
done

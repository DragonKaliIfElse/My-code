banana = $(ps aux | grep '[b]anana' | awk '{print $2}')
echo $banana
kill -9 "$banana"

while true; do
  sleep 1
  OUT=$(vastai show instances)

  if ! echo "$OUT" | grep -q "loading"; then
    echo 'acabou'
    break
  fi
done

#!/bin/sh
# Wait until this machine holds an IP address, for at most a number of seconds.
# Usage: wait-address.sh ADDRESS SECONDS
#
# The units run it before the nagaya starts. A user unit cannot order itself after the system unit tailscaled.service,
# so at boot the tailnet address can be missing. Then the bind fails and exit 3 keeps the unit down (spec 10.4).
set -eu

address="$1"
seconds="$2"

i=0
while [ "$i" -lt "$seconds" ]; do
  if ip -o addr show | grep -q " $address/"; then
    exit 0
  fi
  sleep 1
  i=$((i + 1))
done
echo "the address $address is not on this machine after $seconds s; check tailscaled with \`tailscale ip\`" >&2
exit 1

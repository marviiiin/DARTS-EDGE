#!/bin/bash
# Helper script for API calls with proper auth
ACTION=$1
DIRECTION=$2
HIGHWAY=$3

COOKIE_FILE=$(mktemp)
curl -s -c "$COOKIE_FILE" -X POST "http://127.0.0.1:4430/login" \
    -d "username=admin&password=usfsumlab" > /dev/null

case "$ACTION" in
    start)
        curl -s -b "$COOKIE_FILE" -X POST "http://127.0.0.1:4430/start-detection" \
            -H "Content-Type: application/json" \
            -d "{\"droneDirection\":\"$DIRECTION\",\"highwayName\":\"$HIGHWAY\",\"videoDuration\":300}"
        ;;
    stop)
        curl -s -b "$COOKIE_FILE" -X POST "http://127.0.0.1:4430/stop-detection"
        ;;
    status)
        curl -s -b "$COOKIE_FILE" "http://127.0.0.1:4430/get-live-traffic-status"
        ;;
esac

rm -f "$COOKIE_FILE"

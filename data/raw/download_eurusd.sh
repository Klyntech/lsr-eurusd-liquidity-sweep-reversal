#!/bin/bash
set -e
cd /home/z/my-project/lsr_eh1/data/raw

# Define quarterly date ranges
ranges=(
    "2019-01-01 2019-04-01"
    "2019-04-01 2019-07-01"
    "2019-07-01 2019-10-01"
    "2019-10-01 2020-01-01"
    "2020-01-01 2020-04-01"
    "2020-04-01 2020-07-01"
    "2020-07-01 2020-10-01"
    "2020-10-01 2021-01-01"
    "2021-01-01 2021-04-01"
    "2021-04-01 2021-07-01"
    "2021-07-01 2021-10-01"
    "2021-10-01 2022-01-01"
    "2022-01-01 2022-04-01"
    "2022-04-01 2022-07-01"
    "2022-07-01 2022-10-01"
    "2022-10-01 2023-01-01"
    "2023-01-01 2023-04-01"
    "2023-04-01 2023-07-01"
    "2023-07-01 2023-10-01"
    "2023-10-01 2024-01-01"
    "2024-01-01 2024-04-01"
    "2024-04-01 2024-07-01"
    "2024-07-01 2024-10-01"
    "2024-10-01 2025-01-01"
    "2025-01-01 2025-04-01"
    "2025-04-01 2025-07-01"
    "2025-07-01 2025-10-01"
    "2025-10-01 2026-01-01"
    "2026-01-01 2026-07-01"
)

for range in "${ranges[@]}"; do
    from=$(echo $range | cut -d' ' -f1)
    to=$(echo $range | cut -d' ' -f2)
    
    # Check if already downloaded successfully
    outfile="download/eurusd-h1-bid-${from}-${to}.csv"
    if [ -f "$outfile" ] && [ $(wc -l < "$outfile") -gt 500 ]; then
        echo "SKIP $from -> $to (exists, $(wc -l < "$outfile") rows)"
        continue
    fi
    
    echo "[$(date +%H:%M:%S)] Downloading $from -> $to ..."
    
    # Download with retry
    timeout 180 npx dukascopy-node \
        -i eurusd \
        -from "$from" \
        -to "$to" \
        -t h1 \
        -p bid \
        -f csv \
        -s \
        -r 3 \
        -rp 2000 \
        -re \
        2>&1
    
    # Find the output file (dukascopy-node names it automatically)
    latest=$(ls -t download/eurusd-h1-bid-*.csv 2>/dev/null | head -1)
    if [ -f "$latest" ] && [ $(wc -l < "$latest") -gt 500 ]; then
        # Rename to our standard name
        mv "$latest" "$outfile"
        echo "  -> OK: $(wc -l < "$outfile") rows"
    else
        echo "  -> FAILED (file too small or missing)"
    fi
    
    # Pause between downloads to be respectful
    sleep 2
done

echo ""
echo "=== Download complete ==="

#!/usr/bin/env python3
"""Inject analysis data into HTML file"""

import json
import re

# Load data
with open('/home/ubuntu/wig30_data/deep_analysis.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Remove chart data from all_stocks to reduce size for non-top stocks
# Keep chart data only for top 7
for i, stock in enumerate(data['all_stocks']):
    if i >= 7:
        # Keep minimal chart data
        if 'chart_data' in stock:
            stock['chart_data'] = {
                'dates': stock['chart_data']['dates'][-20:],
                'close': stock['chart_data']['close'][-20:],
                'volume': stock['chart_data']['volume'][-20:],
                'ma5': stock['chart_data']['ma5'][-20:],
                'ma20': stock['chart_data']['ma20'][-20:],
                'bb_upper': stock['chart_data']['bb_upper'][-20:],
                'bb_lower': stock['chart_data']['bb_lower'][-20:],
                'rsi': stock['chart_data']['rsi'][-20:],
            }

# Convert to JSON string
data_json = json.dumps(data, ensure_ascii=False, separators=(',', ':'))

# Read HTML
with open('/home/ubuntu/wig30_data/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Replace placeholder
html = html.replace('PLACEHOLDER_DATA', data_json)

# Write output
with open('/home/ubuntu/wig30_data/index.html', 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Data injected successfully!")
print(f"Total stocks: {data['total_analyzed']}")
print(f"Strong buys: {len(data['strong_buys'])}")
print(f"Buys: {len(data['buys'])}")
print(f"File size: {len(html) / 1024:.1f} KB")

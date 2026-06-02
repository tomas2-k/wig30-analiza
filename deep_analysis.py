#!/usr/bin/env python3
"""
Deep Technical Analysis for WIG30 Short-term Trading
Algorytm scoringowy v3 - poprawiony stoch_rising + logika AT
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

WIG30_STOCKS = {
    'PKN.WA': {'name': 'PKN Orlen', 'sector': 'Energia', 'ticker': 'PKN'},
    'PKO.WA': {'name': 'PKO Bank Polski', 'sector': 'Finanse', 'ticker': 'PKO'},
    'SPL.WA': {'name': 'Santander Bank Polska', 'sector': 'Finanse', 'ticker': 'SPL'},
    'PEO.WA': {'name': 'Pekao', 'sector': 'Finanse', 'ticker': 'PEO'},
    'KGH.WA': {'name': 'KGHM Polska Miedź', 'sector': 'Górnictwo', 'ticker': 'KGH'},
    'PZU.WA': {'name': 'PZU', 'sector': 'Ubezpieczenia', 'ticker': 'PZU'},
    'MBK.WA': {'name': 'mBank', 'sector': 'Finanse', 'ticker': 'MBK'},
    'LPP.WA': {'name': 'LPP', 'sector': 'Handel detaliczny', 'ticker': 'LPP'},
    'DNP.WA': {'name': 'Dino Polska', 'sector': 'Handel detaliczny', 'ticker': 'DNP'},
    'ALE.WA': {'name': 'Allegro', 'sector': 'E-commerce', 'ticker': 'ALE'},
    'CDR.WA': {'name': 'CD Projekt', 'sector': 'Technologia', 'ticker': 'CDR'},
    'BDX.WA': {'name': 'Budimex', 'sector': 'Budownictwo', 'ticker': 'BDX'},
    'ACP.WA': {'name': 'Asseco Poland', 'sector': 'Technologia', 'ticker': 'ACP'},
    'KRU.WA': {'name': 'Kruk', 'sector': 'Finanse', 'ticker': 'KRU'},
    'MIL.WA': {'name': 'Bank Millennium', 'sector': 'Finanse', 'ticker': 'MIL'},
    'OPL.WA': {'name': 'Orange Polska', 'sector': 'Telekomunikacja', 'ticker': 'OPL'},
    'PGE.WA': {'name': 'PGE', 'sector': 'Energetyka', 'ticker': 'PGE'},
    'ALR.WA': {'name': 'Alior Bank', 'sector': 'Finanse', 'ticker': 'ALR'},
    'JSW.WA': {'name': 'JSW', 'sector': 'Górnictwo', 'ticker': 'JSW'},
    'KTY.WA': {'name': 'Grupa Kęty', 'sector': 'Przemysł', 'ticker': 'KTY'},
    'MDVP.WA': {'name': 'Modivo (CCC)', 'sector': 'Handel detaliczny', 'ticker': 'MDV'},
    'TPE.WA': {'name': 'Tauron', 'sector': 'Energetyka', 'ticker': 'TPE'},
    'ENA.WA': {'name': 'Enea', 'sector': 'Energetyka', 'ticker': 'ENA'},
    'BFT.WA': {'name': 'Benefit Systems', 'sector': 'Usługi', 'ticker': 'BFT'},
    'CPS.WA': {'name': 'Cyfrowy Polsat', 'sector': 'Telekomunikacja', 'ticker': 'CPS'},
    'LWB.WA': {'name': 'Bogdanka', 'sector': 'Górnictwo', 'ticker': 'LWB'},
    'EUR.WA': {'name': 'Eurocash', 'sector': 'Handel detaliczny', 'ticker': 'EUR'},
    'ATT.WA': {'name': 'Grupa Azoty', 'sector': 'Chemia', 'ticker': 'ATT'},
    'MDV.WA': {'name': 'Modivo', 'sector': 'Handel detaliczny', 'ticker': 'MDV2'},
}

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    return sma + (std_dev * std), sma, sma - (std_dev * std)

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
    d = k.rolling(d_period).mean()
    return k, d

def analyze_stock_deep(ticker_wa, info):
    try:
        stock = yf.Ticker(ticker_wa)
        hist = stock.history(period="6mo")

        if hist.empty or len(hist) < 40:
            return None

        close = hist['Close']
        volume = hist['Volume']
        high = hist['High']
        low = hist['Low']
        open_price = hist['Open']

        current_price = float(close.iloc[-1])
        prev_price = float(close.iloc[-2])

        # Price changes
        price_change_1d = ((current_price - prev_price) / prev_price) * 100
        price_change_3d = ((current_price - float(close.iloc[-4])) / float(close.iloc[-4])) * 100 if len(close) >= 4 else 0
        price_change_5d = ((current_price - float(close.iloc[-6])) / float(close.iloc[-6])) * 100 if len(close) >= 6 else 0
        price_change_20d = ((current_price - float(close.iloc[-21])) / float(close.iloc[-21])) * 100 if len(close) >= 21 else 0
        price_change_60d = ((current_price - float(close.iloc[-61])) / float(close.iloc[-61])) * 100 if len(close) >= 61 else 0

        # RSI
        rsi = calculate_rsi(close)
        current_rsi = float(rsi.iloc[-1])
        rsi_prev = float(rsi.iloc[-2])
        rsi_3d_ago = float(rsi.iloc[-4]) if len(rsi) >= 4 else rsi_prev
        rsi_rising = current_rsi > rsi_prev
        rsi_recovering = current_rsi > rsi_3d_ago

        # MACD
        macd_line, signal_line, histogram = calculate_macd(close)
        current_hist = float(histogram.iloc[-1])
        prev_hist = float(histogram.iloc[-2])
        prev2_hist = float(histogram.iloc[-3])
        macd_crossover = current_hist > 0 and prev_hist <= 0
        macd_improving = current_hist > prev_hist
        macd_improving_3d = current_hist > prev2_hist

        # Bollinger Bands
        bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close)
        bb_pos_denom = float(bb_upper.iloc[-1]) - float(bb_lower.iloc[-1])
        bb_position = ((current_price - float(bb_lower.iloc[-1])) / bb_pos_denom * 100) if bb_pos_denom > 0 else 50
        bb_width = (bb_pos_denom / float(bb_mid.iloc[-1])) * 100 if float(bb_mid.iloc[-1]) > 0 else 0

        # Stochastic - POPRAWKA: .iloc[-1] zamiast całej serii
        stoch_k, stoch_d = calculate_stochastic(high, low, close)
        current_stoch_k = float(stoch_k.iloc[-1])
        current_stoch_d = float(stoch_d.iloc[-1])
        stoch_rising = current_stoch_k > current_stoch_d  # teraz porównuje float do float

        # Volume
        avg_volume_20d = float(volume.iloc[-21:-1].mean())
        current_volume = float(volume.iloc[-1])
        volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else 1

        # Moving averages
        ma5 = float(close.rolling(5).mean().iloc[-1])
        ma10 = float(close.rolling(10).mean().iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else ma20
        ma5_prev = float(close.rolling(5).mean().iloc[-2])
        ma20_prev = float(close.rolling(20).mean().iloc[-2])

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        atr_pct = (atr / current_price) * 100

        # 52-week high/low
        week52_high = float(high.max())
        week52_low = float(low.min())
        distance_from_high = ((current_price - week52_high) / week52_high) * 100
        distance_from_low = ((current_price - week52_low) / week52_low) * 100

        # Candle
        last_open = float(open_price.iloc[-1])
        last_high = float(high.iloc[-1])
        last_low = float(low.iloc[-1])
        last_candle_body = abs(current_price - last_open)
        last_candle_range = last_high - last_low if last_high > last_low else 0.0001
        last_candle_type = "bullish" if current_price > last_open else "bearish"

        # ============================================================
        # ALGORYTM SCORINGOWY v3
        # Zgodny ze sztuką analizy technicznej:
        # - RSI wymaga potwierdzenia kierunku
        # - Wykupienie (RSI>75, BB>90%) = UNIKAJ niezależnie
        # - MACD crossover = najsilniejszy sygnał
        # - Wolumen potwierdza ruch
        # - Trend (MA) = filtr strukturalny
        # ============================================================
        score = 0
        signals = []
        risk_factors = []

        # === 1. TREND 5-DNIOWY ===
        if price_change_5d < -6:
            score -= 3
            risk_factors.append(f"Silny trend spadkowy 5D ({price_change_5d:.1f}%)")
        elif price_change_5d < -3:
            score -= 1
            risk_factors.append(f"Korekta 5D ({price_change_5d:.1f}%)")
        elif price_change_5d > 3:
            score += 1
            signals.append(f"Trend wzrostowy 5D (+{price_change_5d:.1f}%)")

        # === 2. RSI - wymaga potwierdzenia kierunku ===
        if current_rsi < 25:
            if rsi_rising and rsi_recovering:
                score += 2
                signals.append(f"RSI ekstremalnie wyprzedany ({current_rsi:.1f}) + odbicie")
            else:
                score -= 3
                risk_factors.append(f"RSI ekstremalnie wyprzedany ({current_rsi:.1f}) i spada - niebezpieczne")
        elif current_rsi < 35:
            if rsi_rising and rsi_recovering:
                score += 3
                signals.append(f"RSI wyprzedany ({current_rsi:.1f}) + rosnący - sygnał odbicia")
            elif rsi_rising:
                score += 1
                signals.append(f"RSI wyprzedany ({current_rsi:.1f}) - ostrożnie")
            else:
                score -= 2
                risk_factors.append(f"RSI wyprzedany ({current_rsi:.1f}) ale spada - brak dna")
        elif current_rsi < 50:
            if rsi_rising:
                score += 2
                signals.append(f"RSI wychodzi ze strefy wyprzedania ({current_rsi:.1f})")
            else:
                score += 0
        elif current_rsi < 65:
            if rsi_rising:
                score += 2
                signals.append(f"RSI w strefie byczej ({current_rsi:.1f}) - momentum")
            else:
                score += 1
                signals.append(f"RSI neutralny ({current_rsi:.1f})")
        elif current_rsi < 75:
            # Strefa 65-75: spółka rośnie, nie jest jeszcze wykupiona
            score += 0
        else:
            # RSI >= 75: wykupiony - wyraźna kara
            score -= 3
            risk_factors.append(f"RSI wykupiony ({current_rsi:.1f}) - wysoke ryzyko korekty")

        # === 3. MACD - najsilniejszy sygnał ===
        if macd_crossover:
            score += 4
            signals.append("MACD bullish crossover - silny sygnał kupna!")
        elif macd_improving and current_hist > 0:
            score += 2
            signals.append("MACD momentum pozytywny i rosnący")
        elif macd_improving_3d and current_hist < 0:
            if rsi_rising:
                score += 2
                signals.append("MACD wychodzi z dołka + RSI odbija")
            else:
                score += 1
                signals.append("MACD histogram poprawia się")
        elif not macd_improving:
            score -= 1
            risk_factors.append("MACD słabnie")

        # === 4. BOLLINGER BANDS ===
        if bb_position < 5:
            if rsi_rising or price_change_3d > 0:
                score += 3
                signals.append(f"Cena przy dolnym BB ({bb_position:.0f}%) + odbicie")
            else:
                score += 0
                risk_factors.append(f"Cena przy dolnym BB ({bb_position:.0f}%) bez odbicia")
        elif bb_position < 20:
            score += 2
            signals.append(f"Cena blisko dolnego BB ({bb_position:.0f}%) - wsparcie")
        elif bb_position < 40:
            score += 1
            signals.append(f"Cena poniżej środka BB ({bb_position:.0f}%)")
        elif bb_position > 95:
            # Powyżej górnego BB = wykupienie, silna kara
            score -= 3
            risk_factors.append(f"Cena powyżej górnego BB ({bb_position:.0f}%) - wykupienie")
        elif bb_position > 80:
            score -= 1
            risk_factors.append(f"Cena blisko górnego BB ({bb_position:.0f}%) - opór")

        # === 5. STOCHASTIC ===
        if current_stoch_k < 20:
            if stoch_rising and current_stoch_k > current_stoch_d:
                score += 2
                signals.append(f"Stochastic wyprzedany + crossover ({current_stoch_k:.0f})")
            # bez potwierdzenia = neutralne
        elif current_stoch_k < 35:
            score += 1
            signals.append(f"Stochastic w strefie wyprzedania ({current_stoch_k:.0f})")
        elif current_stoch_k > 85:
            score -= 1
            risk_factors.append(f"Stochastic wykupiony ({current_stoch_k:.0f})")

        # === 6. ŚREDNIE KROCZĄCE (trend strukturalny) ===
        if current_price > ma5 > ma10 > ma20:
            score += 3
            signals.append("Cena powyżej MA5>MA10>MA20 - silny uptrend")
        elif current_price > ma5 and ma5 > ma20:
            score += 2
            signals.append("Cena powyżej MA5>MA20 - uptrend")
        elif current_price > ma20:
            score += 1
            signals.append("Cena powyżej MA20 - trend wzrostowy")
        elif current_price < ma50:
            score -= 1
            risk_factors.append("Cena poniżej MA50 - trend spadkowy")

        # Golden cross MA5/MA20
        if ma5 > ma20 and ma5_prev <= ma20_prev:
            score += 2
            signals.append("Golden Cross MA5/MA20 - sygnał kupna")

        # === 7. WOLUMEN ===
        if volume_ratio > 2.0 and (rsi_rising or price_change_3d > 0):
            score += 3
            signals.append(f"Bardzo wysoki wolumen ({volume_ratio:.1f}x) + wzrost - potwierdzenie")
        elif volume_ratio > 1.5:
            score += 1
            signals.append(f"Podwyższony wolumen ({volume_ratio:.1f}x)")
        elif volume_ratio < 0.4:
            score -= 1
            risk_factors.append("Bardzo niski wolumen - brak przekonania")

        # === 8. ATR (zmienność) ===
        if 2.0 <= atr_pct <= 4.0:
            score += 1
            signals.append(f"Dobra zmienność dla celu 2-3% (ATR {atr_pct:.1f}%)")
        elif atr_pct < 1.5:
            score -= 1
            risk_factors.append(f"Niska zmienność (ATR {atr_pct:.1f}%) - trudno osiągnąć cel")

        # === 9. ŚWIECA ===
        if last_candle_type == "bullish" and last_candle_body > last_candle_range * 0.6:
            score += 1
            signals.append("Silna bycza świeca")
        elif last_candle_type == "bearish" and last_candle_body > last_candle_range * 0.7:
            score -= 1
            risk_factors.append("Silna niedźwiedzia świeca")

        # === 10. ODLEGŁOŚĆ OD SZCZYTU ===
        if -20 <= distance_from_high <= -5:
            score += 1
            signals.append(f"Korekta od szczytu ({distance_from_high:.1f}%) - potencjał powrotu")
        elif distance_from_high < -40:
            score -= 1
            risk_factors.append(f"Daleko od szczytu ({distance_from_high:.1f}%)")

        # ============================================================
        # REKOMENDACJA
        # ============================================================
        if score >= 11:
            recommendation = "MOCNE KUPNO"
            rec_color = "#00c853"
            rec_icon = "🟢"
        elif score >= 8:
            recommendation = "KUPNO"
            rec_color = "#69f0ae"
            rec_icon = "🟢"
        elif score >= 5:
            recommendation = "UMIARKOWANE KUPNO"
            rec_color = "#ffeb3b"
            rec_icon = "🟡"
        elif score >= 2:
            recommendation = "OBSERWUJ"
            rec_color = "#ff9800"
            rec_icon = "🟠"
        else:
            recommendation = "UNIKAJ"
            rec_color = "#f44336"
            rec_icon = "🔴"

        # Targets
        target_2pct = current_price * 1.02
        target_25pct = current_price * 1.025
        target_3pct = current_price * 1.03
        stop_loss = current_price * 0.985

        # Probability
        bullish_signals = len(signals)
        bearish_signals = len(risk_factors)
        total_signals = bullish_signals + bearish_signals
        probability = (bullish_signals / total_signals * 100) if total_signals > 0 else 50

        # Chart data
        chart_data_close = close.tail(60).round(2).tolist()
        chart_data_dates = [d.strftime('%Y-%m-%d') for d in close.tail(60).index]
        chart_data_volume = volume.tail(60).tolist()
        chart_data_ma5 = close.rolling(5).mean().tail(60).round(2).tolist()
        chart_data_ma20 = close.rolling(20).mean().tail(60).round(2).tolist()
        chart_data_bb_upper = bb_upper.tail(60).round(2).tolist()
        chart_data_bb_lower = bb_lower.tail(60).round(2).tolist()
        chart_data_rsi = rsi.tail(60).round(1).tolist()

        result = {
            'ticker': info['ticker'],
            'ticker_wa': ticker_wa,
            'name': info['name'],
            'sector': info['sector'],
            'current_price': round(current_price, 2),
            'price_change_1d': round(price_change_1d, 2),
            'price_change_3d': round(price_change_3d, 2),
            'price_change_5d': round(price_change_5d, 2),
            'price_change_20d': round(price_change_20d, 2),
            'price_change_60d': round(price_change_60d, 2),
            'rsi': round(current_rsi, 1),
            'rsi_trend': "rosnący" if rsi_rising else "malejący",
            'macd_hist': round(current_hist, 4),
            'macd_crossover': bool(macd_crossover),
            'bb_position': round(bb_position, 1),
            'bb_width': round(bb_width, 1),
            'stoch_k': round(current_stoch_k, 1),
            'stoch_d': round(current_stoch_d, 1),
            'volume_ratio': round(volume_ratio, 2),
            'atr_pct': round(atr_pct, 2),
            'ma5': round(ma5, 2),
            'ma10': round(ma10, 2),
            'ma20': round(ma20, 2),
            'ma50': round(ma50, 2),
            'week52_high': round(week52_high, 2),
            'week52_low': round(week52_low, 2),
            'distance_from_high': round(distance_from_high, 1),
            'distance_from_low': round(distance_from_low, 1),
            'score': score,
            'signals': signals,
            'risk_factors': risk_factors,
            'recommendation': recommendation,
            'rec_color': rec_color,
            'rec_icon': rec_icon,
            'target_2pct': round(target_2pct, 2),
            'target_25pct': round(target_25pct, 2),
            'target_3pct': round(target_3pct, 2),
            'stop_loss': round(stop_loss, 2),
            'probability': round(probability, 0),
            'chart_data': {
                'dates': chart_data_dates,
                'close': chart_data_close,
                'volume': chart_data_volume,
                'ma5': chart_data_ma5,
                'ma20': chart_data_ma20,
                'bb_upper': chart_data_bb_upper,
                'bb_lower': chart_data_bb_lower,
                'rsi': chart_data_rsi,
            }
        }

        print(f"  {info['ticker']:6s} | Score: {score:3d} | RSI: {current_rsi:5.1f} | {recommendation:20s} | {info['name']}")
        return result

    except Exception as e:
        print(f"  {ticker_wa}: Error - {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("=" * 80)
    print("WIG30 ANALIZA KRÓTKOTERMINOWA - OKAZJE INWESTYCYJNE (1-3 dni, cel 2-3%)")
    print(f"Data analizy: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    print("=" * 80)

    results = []
    for ticker_wa, info in WIG30_STOCKS.items():
        result = analyze_stock_deep(ticker_wa, info)
        if result:
            results.append(result)

    results.sort(key=lambda x: x['score'], reverse=True)

    strong_buys = [r for r in results if r['score'] >= 11]
    buys = [r for r in results if 8 <= r['score'] < 11]
    moderate_buys = [r for r in results if 5 <= r['score'] < 8]
    watch = [r for r in results if 2 <= r['score'] < 5]
    avoid = [r for r in results if r['score'] < 2]

    print("\n" + "=" * 80)
    print("PODSUMOWANIE WYNIKÓW")
    print("=" * 80)
    print(f"Przeanalizowano: {len(results)} spółek")
    print(f"Mocne kupno:     {len(strong_buys)} spółek")
    print(f"Kupno:           {len(buys)} spółek")
    print(f"Umiarkowane:     {len(moderate_buys)} spółek")
    print(f"Obserwuj:        {len(watch)} spółek")
    print(f"Unikaj:          {len(avoid)} spółek")

    print("\nTOP 10 OKAZJI:")
    print("-" * 80)
    for r in results[:10]:
        print(f"  {r['ticker']:6s} | {r['name']:25s} | Score: {r['score']:2d} | RSI: {r['rsi']:5.1f} | {r['recommendation']}")
        print(f"         Cena: {r['current_price']:.2f} PLN | Cel: {r['target_2pct']:.2f}-{r['target_3pct']:.2f} | SL: {r['stop_loss']:.2f}")
        for sig in r['signals'][:2]:
            print(f"         ✓ {sig}")
        if r['risk_factors']:
            print(f"         ⚠ {r['risk_factors'][0]}")
        print()

    sector_scores = {}
    for r in results:
        sector = r['sector']
        if sector not in sector_scores:
            sector_scores[sector] = []
        sector_scores[sector].append(r['score'])

    sector_avg = {s: round(np.mean(v), 1) for s, v in sector_scores.items()}
    sector_avg_sorted = dict(sorted(sector_avg.items(), key=lambda x: x[1], reverse=True))

    output = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'analysis_date_display': datetime.now().strftime('%d.%m.%Y %H:%M'),
        'total_analyzed': len(results),
        'strong_buys': strong_buys,
        'buys': buys,
        'moderate_buys': moderate_buys,
        'watch': watch,
        'avoid': avoid,
        'all_stocks': results,
        'sector_scores': sector_avg_sorted,
        'top_picks': results[:5],
    }

    with open('data/deep_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nWyniki zapisane do data/deep_analysis.json")
    return output

if __name__ == '__main__':
    main()

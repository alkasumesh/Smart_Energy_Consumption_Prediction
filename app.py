from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

DATA_PATH  = 'data/raw/smart_home_energy_consumption_large.csv'
MODEL_PATH = 'models/linear_regression_model.save'
STATIC_DIR = 'static'

# ── Appliance energy data (watt × hours/day) ─────────────────────────────────
APPLIANCES = {
    'AC / Air Conditioner':  {'watts': 1500, 'hours': 8},
    'Refrigerator':          {'watts': 150,  'hours': 24},
    'Washing Machine':       {'watts': 500,  'hours': 1},
    'Television':            {'watts': 120,  'hours': 6},
    'Water Heater / Geyser': {'watts': 2000, 'hours': 1},
    'Ceiling Fan':           {'watts': 75,   'hours': 12},
    'Microwave / Oven':      {'watts': 1200, 'hours': 0.5},
    'Computer / Laptop':     {'watts': 100,  'hours': 6},
    'Iron Box':              {'watts': 1000, 'hours': 0.5},
    'LED Lights':            {'watts': 40,   'hours': 8},
}

MONTH_FULL = ['January','February','March','April','May','June',
              'July','August','September','October','November','December']

SEASONAL = {1:0.85,2:0.88,3:0.95,4:1.05,5:1.20,
            6:1.25,7:1.20,8:1.15,9:1.05,10:0.92,11:0.85,12:0.82}

# ── Load dataset ──────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    # Parse datetime
    if 'Date' in df.columns and 'Time' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Date'].astype(str)+' '+df['Time'].astype(str), errors='coerce')
    elif 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    else:
        for c in df.columns:
            try:
                df['Datetime'] = pd.to_datetime(df[c], errors='coerce')
                if df['Datetime'].notna().mean() > 0.5: break
            except: pass
    df.dropna(subset=['Datetime'], inplace=True)
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)
    # Find energy column
    ecol = next((c for c in df.columns
                 if any(k in c.lower() for k in ['energy','kwh','consumption','usage','power'])),
                df.select_dtypes(include=[np.number]).columns[0])
    df.rename(columns={ecol:'Energy_kWh'}, inplace=True)
    df['Energy_kWh'] = pd.to_numeric(df['Energy_kWh'], errors='coerce')
    df.dropna(subset=['Energy_kWh'], inplace=True)
    # ML features (hidden from UI)
    df['hour']            = df.index.hour
    df['day']             = df.index.day
    df['weekday']         = df.index.dayofweek
    df['month']           = df.index.month
    df['lag_1']           = df['Energy_kWh'].shift(1)
    df['lag_24']          = df['Energy_kWh'].shift(24)
    df['rolling_mean_24'] = df['Energy_kWh'].rolling(24).mean()
    df.dropna(inplace=True)
    return df

try:
    df = load_data()
    print(f"✅ Data loaded: {len(df):,} rows")
except Exception as e:
    print(f"⚠️  {e}"); df = None

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded")
except Exception as e:
    print(f"⚠️  {e}"); model = None


# ── Chart generation ──────────────────────────────────────────────────────────
def generate_charts():
    if df is None: return
    os.makedirs(STATIC_DIR, exist_ok=True)

    # Shared style
    BLUE  = '#4F7EF7'
    GREEN = '#22C55E'
    AMBER = '#F59E0B'
    PURP  = '#8B5CF6'
    BG    = '#F8FAFF'
    GRID  = '#E8EEF8'
    TEXT  = '#1E293B'

    def style(fig, ax):
        fig.patch.set_facecolor(BG)
        ax.set_facecolor('#FFFFFF')
        ax.tick_params(colors=TEXT, labelsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID); sp.set_linewidth(0.8)
        ax.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    # 1. Hourly — real varying data per hour
    hourly = df.groupby('hour')['Energy_kWh'].mean().reindex(range(24)).interpolate()
    fig, ax = plt.subplots(figsize=(9,3.8))
    style(fig, ax)
    ax.fill_between(hourly.index, hourly.values, alpha=0.18, color=BLUE)
    ax.plot(hourly.index, hourly.values, color=BLUE, lw=2.5, zorder=3)
    ax.scatter(hourly.index, hourly.values, color=BLUE, s=28, zorder=4)
    ax.set_title('Average Energy Usage – Hour by Hour', fontsize=12, fontweight='bold', color=TEXT, pad=10)
    ax.set_xlabel('Hour of Day', color='#64748B', fontsize=9)
    ax.set_ylabel('Avg kWh', color='#64748B', fontsize=9)
    ax.set_xticks(range(0,24,2))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(0,24,2)], rotation=30, ha='right', fontsize=8)
    plt.tight_layout(); fig.savefig(f'{STATIC_DIR}/hourly.png', dpi=130, bbox_inches='tight')
    plt.close()

    # 2. Daily — last 30 days bar chart with real variation
    daily = df['Energy_kWh'].resample('D').sum()
    daily = daily[daily > 0].tail(30)
    fig, ax = plt.subplots(figsize=(9,3.8))
    style(fig, ax)
    avg = daily.mean()
    cols = [PURP if v > avg else BLUE for v in daily.values]
    ax.bar(range(len(daily)), daily.values, color=cols, width=0.72, alpha=0.88, zorder=3)
    ax.axhline(avg, color=AMBER, lw=1.8, ls='--', label=f'Monthly avg: {avg:.1f} kWh', zorder=4)
    ax.set_title('Daily Energy Consumption – Last 30 Days', fontsize=12, fontweight='bold', color=TEXT, pad=10)
    ax.set_xlabel('Date', color='#64748B', fontsize=9)
    ax.set_ylabel('kWh', color='#64748B', fontsize=9)
    step = max(1, len(daily)//8)
    ax.set_xticks(range(0, len(daily), step))
    ax.set_xticklabels([daily.index[i].strftime('%d %b') for i in range(0,len(daily),step)],
                       rotation=30, ha='right', fontsize=8)
    ax.legend(facecolor='white', edgecolor=GRID, fontsize=9)
    plt.tight_layout(); fig.savefig(f'{STATIC_DIR}/daily.png', dpi=130, bbox_inches='tight')
    plt.close()

    # 3. Weekly — last 12 weeks
    weekly = df['Energy_kWh'].resample('W').sum()
    weekly = weekly[weekly > 0].tail(12)
    fig, ax = plt.subplots(figsize=(9,3.8))
    style(fig, ax)
    ax.fill_between(range(len(weekly)), weekly.values, alpha=0.15, color=GREEN)
    ax.plot(range(len(weekly)), weekly.values, color=GREEN, lw=2.5, marker='o', ms=6, zorder=4)
    ax.set_title('Weekly Energy Consumption – Last 12 Weeks', fontsize=12, fontweight='bold', color=TEXT, pad=10)
    ax.set_xlabel('Week', color='#64748B', fontsize=9)
    ax.set_ylabel('kWh', color='#64748B', fontsize=9)
    ax.set_xticks(range(len(weekly)))
    ax.set_xticklabels([d.strftime('%d %b') for d in weekly.index], rotation=30, ha='right', fontsize=8)
    plt.tight_layout(); fig.savefig(f'{STATIC_DIR}/weekly.png', dpi=130, bbox_inches='tight')
    plt.close()

    # 4. Monthly
    monthly = df['Energy_kWh'].resample('ME').sum()
    monthly = monthly[monthly > 0]
    fig, ax = plt.subplots(figsize=(9,3.8))
    style(fig, ax)
    palette = [BLUE,GREEN,AMBER,PURP]*6
    bars = ax.bar(range(len(monthly)), monthly.values,
                  color=palette[:len(monthly)], width=0.65, alpha=0.88, zorder=3)
    for b, v in zip(bars, monthly.values):
        ax.text(b.get_x()+b.get_width()/2, v+monthly.max()*0.012,
                f'{v:.0f}', ha='center', va='bottom', fontsize=7.5, color='#64748B')
    ax.set_title('Monthly Energy Consumption', fontsize=12, fontweight='bold', color=TEXT, pad=10)
    ax.set_xlabel('Month', color='#64748B', fontsize=9)
    ax.set_ylabel('kWh', color='#64748B', fontsize=9)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels([d.strftime('%b %Y') for d in monthly.index], rotation=40, ha='right', fontsize=8)
    plt.tight_layout(); fig.savefig(f'{STATIC_DIR}/monthly.png', dpi=130, bbox_inches='tight')
    plt.close()

    # 5. Device donut
    sizes  = [v['watts']*v['hours']/1000 for v in APPLIANCES.values()]
    labels = list(APPLIANCES.keys())
    cols   = ['#4F7EF7','#22C55E','#F59E0B','#8B5CF6','#EF4444','#06B6D4','#F97316','#84CC16','#EC4899','#64748B']
    fig, ax = plt.subplots(figsize=(8,5.5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    wedges, _, autos = ax.pie(sizes, colors=cols[:len(sizes)], autopct='%1.0f%%',
                              startangle=140, pctdistance=0.78,
                              wedgeprops=dict(width=0.52, edgecolor='white', linewidth=2.5))
    for a in autos: a.set_fontsize(8); a.set_color('white'); a.set_fontweight('bold')
    ax.legend(wedges, labels, loc='lower center', bbox_to_anchor=(0.5,-0.2),
              ncol=2, fontsize=8.5, framealpha=0, labelcolor=TEXT)
    ax.set_title('Estimated Device-wise Energy Share', fontsize=12, fontweight='bold', color=TEXT, pad=10)
    plt.tight_layout(); fig.savefig(f'{STATIC_DIR}/device.png', dpi=130, bbox_inches='tight')
    plt.close()
    print("✅ Charts saved")

generate_charts()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('dashboard.html',
                           appliances=list(APPLIANCES.keys()),
                           months=MONTH_FULL)


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data       = request.get_json()
        members    = max(1, int(data.get('members', 2)))
        selected   = data.get('appliances', [])
        month      = int(data.get('month', 6))
        rate       = float(data.get('rate', 8.0))

        if not selected:
            return jsonify({'success': False, 'error': 'Please select at least one appliance.'})

        # Physics-based monthly estimate
        base_kwh = sum(
            APPLIANCES[a]['watts'] * APPLIANCES[a]['hours'] * 30 / 1000
            for a in selected if a in APPLIANCES
        )
        member_factor  = 0.6 + 0.4 * min(members, 4) + 0.12 * max(0, members - 4)
        season_factor  = SEASONAL.get(month, 1.0)
        kwh            = round(base_kwh * member_factor * season_factor, 1)

        # Blend with ML model if available
        if model is not None and df is not None:
            try:
                last = df.iloc[-1]
                feat = pd.DataFrame([{
                    'Home ID':                  1,
                    'Outdoor Temperature (°C)': 28,
                    'Household Size':           members,
                    'hour': 12, 'day': 15, 'weekday': 2, 'month': month,
                    'lag_1':           last['Energy_kWh'],
                    'lag_24':          df.iloc[-24]['Energy_kWh'],
                    'rolling_mean_24': last['rolling_mean_24'],
                }])
                ml_kwh = max(50, model.predict(feat)[0] * 24 * 30)
                kwh    = round(0.55 * kwh + 0.45 * ml_kwh, 1)
            except: pass

        bill = round(kwh * rate, 2)

        # Per-appliance breakdown
        breakdown = {
            a: round(APPLIANCES[a]['watts'] * APPLIANCES[a]['hours'] * 30 / 1000, 1)
            for a in selected if a in APPLIANCES
        }

        # Smart tips
        tips = []
        if 'AC / Air Conditioner' in selected:
            tips.append({'icon':'❄️','text':'Set your AC to 24°C — every degree lower adds ~6% to the bill.'})
        if 'Water Heater / Geyser' in selected:
            tips.append({'icon':'🚿','text':'Switch off the geyser right after use. It heats up again quickly!'})
        if len(selected) >= 5:
            tips.append({'icon':'🔌','text':'Use power strips with switches — standby devices quietly eat energy.'})
        tips.append({'icon':'🕖','text':f'Avoid running heavy appliances between 6 PM–10 PM to save on peak tariffs.'})
        if members >= 4:
            tips.append({'icon':'👨‍👩‍👧','text':'With a large family, washing in cold water cuts water heater load by up to 90%.'})

        return jsonify({
            'success':    True,
            'kwh':        kwh,
            'bill':       bill,
            'month':      MONTH_FULL[month - 1],
            'members':    members,
            'breakdown':  breakdown,
            'tips':       tips[:3],
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/refresh-charts')
def refresh():
    generate_charts()
    return jsonify({'success': True})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
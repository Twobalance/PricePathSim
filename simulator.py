
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import uuid

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
st.set_page_config(layout="wide", page_title="PriceSim", page_icon="📈")

INITIAL_BALANCE = 10000.0
SYMBOL = "XYZ/USDT"
DT = 1/365/24/60/60  # 1 second time step in years (approx) for GBM
MU = 0.1  # Drift
SIGMA = 2.0  # Volatility (Increased for more action)
JUMP_INTENSITY = 0.05 # Higher probability of jump
JUMP_MEAN = 0.0 
JUMP_STD = 0.005 # ~0.5% jumps (approx $250 moves at $50k)

# ==========================================
# CORE CLASSES
# ==========================================

class MarketEngine:
    def __init__(self):
        if 'market_data' not in st.session_state:
            # Initialize with some history
            self.price = 50000.0
            st.session_state.market_data = pd.DataFrame(columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
            self.generate_history(100) # seeded history
        else:
            self.price = st.session_state.market_data.iloc[-1]['Close'] if not st.session_state.market_data.empty else 50000.0
            if not st.session_state.market_data.empty:
                self.last_time = st.session_state.market_data.iloc[-1]['Time']
            else:
                 self.last_time = datetime.now()

    def generate_history(self, periods):
        data = []
        current_p = self.price
        now = datetime.now()
        
        # Go back in time
        start_time = now - pd.Timedelta(seconds=periods)
        
        for i in range(periods):
            # Previous Close is Open
            open_p = current_p
            
            # Simulate "Micro-ticks" within the candle to form High/Low
            high_p = open_p
            low_p = open_p
            temp_p = open_p
            
            # 10 micro-steps per candle
            for _ in range(10):
                drift = (MU - 0.5 * SIGMA**2) * (DT/10)
                shock = SIGMA * np.sqrt(DT/10) * np.random.normal()
                
                # Jump (lower prob per micro-step)
                if np.random.random() < JUMP_INTENSITY/10:
                    jump = np.random.normal(JUMP_MEAN, JUMP_STD)
                    temp_p *= np.exp(jump)
                
                temp_p *= np.exp(drift + shock)
                high_p = max(high_p, temp_p)
                low_p = min(low_p, temp_p)
            
            close_p = temp_p
            current_p = close_p
            
            # Volume based on volatility (range)
            candle_range = abs(high_p - low_p) / open_p
            base_vol = 100 + (candle_range * 1000000) # Arbitrary scale
            volume = base_vol * np.random.uniform(0.8, 1.2)
            
            data.append({
                'Time': start_time + pd.Timedelta(seconds=i),
                'Open': open_p, 'High': high_p, 'Low': low_p, 'Close': close_p,
                'Volume': volume
            })
            
        df = pd.DataFrame(data)
        st.session_state.market_data = df # Replace init
        self.price = current_p
        self.last_time = data[-1]['Time']

    def tick(self):
        # Generate next candle with micro-ticks
        last_close = self.price
        open_p = last_close
        
        # Get dynamic volatility from session state
        current_sigma = st.session_state.get('volatility', SIGMA)
        
        high_p = open_p
        low_p = open_p
        temp_p = open_p
        
        # 10 micro-steps
        for _ in range(10):
            drift = (MU - 0.5 * current_sigma**2) * (DT/10)
            shock = current_sigma * np.sqrt(DT/10) * np.random.normal()
            
            if np.random.random() < JUMP_INTENSITY/10:
                jump = np.random.normal(JUMP_MEAN, JUMP_STD)
                temp_p *= np.exp(jump)
                
            temp_p *= np.exp(drift + shock)
            high_p = max(high_p, temp_p)
            low_p = min(low_p, temp_p)
            
        new_close = temp_p
        self.price = new_close
        
        # Strict Timestamping
        new_time = self.last_time + pd.Timedelta(seconds=1)
        self.last_time = new_time
        
        # Volume
        candle_range = abs(high_p - low_p) / open_p
        base_vol = 100 + (candle_range * 1000000)
        volume = base_vol * np.random.uniform(0.8, 1.2)
        
        new_row = {
            'Time': new_time,
            'Open': open_p,
            'High': high_p,
            'Low': low_p,
            'Close': new_close,
            'Volume': volume
        }
        
        df = pd.DataFrame([new_row])
        st.session_state.market_data = pd.concat([st.session_state.market_data, df], ignore_index=True)
        
        if len(st.session_state.market_data) > 300: # Increase buffer for MA
             st.session_state.market_data = st.session_state.market_data.iloc[-300:]
             
        return self.price

class TradingEngine:
    def __init__(self):
        if 'wallet' not in st.session_state:
            st.session_state.wallet = {
                'balance': INITIAL_BALANCE,
                'used_margin': 0.0,
                'pnl': 0.0,
                'equity': INITIAL_BALANCE
            }
        if 'positions' not in st.session_state:
            st.session_state.positions = []
        
        if 'orders' not in st.session_state:
            st.session_state.orders = []

    def place_order(self, side, amount, leverage, margin_mode, current_price):
        margin_needed = (amount * current_price) / leverage
        
        # Basic check
        available = st.session_state.wallet['balance'] - st.session_state.wallet['used_margin'] 
        # Note: Cross margin logic simplifies availability check to Equity, we stick to balance for simplification or use specific cross logic
        
        # Adjust implementation for Cross/Isolated
        if margin_mode == 'Isolated':
            if margin_needed > available:
                return False, "Insufficient Margin"
        else: # Cross
             # In cross, we look at total equity
             if margin_needed > (st.session_state.wallet['equity'] - st.session_state.wallet['used_margin']):
                  return False, "Insufficient Equity"

        position = {
            'id': str(uuid.uuid4())[:8],
            'symbol': SYMBOL,
            'side': side,
            'size': amount,
            'entry_price': current_price,
            'leverage': leverage,
            'margin_mode': margin_mode,
            'initial_margin': margin_needed,
            'liq_price': 0.0,
            'pnl': 0.0,
            'active': True
        }
        
        # Calculate Liquidation Price
        # Long: Entry * (1 - 1/Lev + MMR)
        # Short: Entry * (1 + 1/Lev - MMR)
        mmr = 0.005 # 0.5% Maintenance Margin
        
        if margin_mode == 'Isolated':
            if side == 'Long':
                position['liq_price'] = current_price * (1 - 1/leverage + mmr)
            else:
                position['liq_price'] = current_price * (1 + 1/leverage - mmr)
        else:
            # Cross Liq is dynamic based on account balance, simplified here to static estimation or updated in loop
            # For this MVP, we will calculate a static approximation for Cross or update it in `update_positions`
            # Approximate for simplicity: share the total balance
            position['liq_price'] = 0.0 # Will calculate dynamically

        st.session_state.positions.append(position)
        st.session_state.wallet['used_margin'] += margin_needed
        return True, "Order Filled"

    def update_positions(self, current_price):
        total_pnl = 0.0
        active_positions = []
        
        wallet = st.session_state.wallet
        equity = wallet['balance'] # Start with static balance
        
        # First Calc PnL
        for pos in st.session_state.positions:
            if not pos['active']: continue
            
            diff = current_price - pos['entry_price']
            if pos['side'] == 'Short':
                diff = -diff
            
            pnl = diff * pos['size']
            pos['pnl'] = pnl
            total_pnl += pnl
        
        # Update Equity
        wallet['pnl'] = total_pnl
        wallet['equity'] = wallet['balance'] + total_pnl
        
        # Check Liquidation
        mmr = 0.005 # Maintenance Margin Requirement

        for pos in st.session_state.positions:
            if not pos['active']: continue
            
            liquidated = False
            
            if pos['margin_mode'] == 'Isolated':
                # Margin Balance = Initial Margin + PnL
                margin_balance = pos['initial_margin'] + pos['pnl']
                # Maintenance Margin Requirement = Position Value * MMR
                maint_margin_req = (pos['size'] * current_price) * mmr
                
                if margin_balance < maint_margin_req:
                    liquidated = True
                    
            else: # Cross
                # Cross Margin Ratio = Equity / Total Maint Margin
                # If Equity < Total Maint Margin -> Liquidate ALL (Simplified) or liquidate largest loser
                # Here we check if Equity is effectively gone or below maintenance for this pos
                # Simplified: If Equity < 0, boom. Or stricter:
                # If Equity < Total Positions Value * MMR
                
                # Check if specific position is dragging account under
                # We often treat Cross as "Account Level Liquidation"
                pass 

        # Global Cross Check
        total_position_value = sum([p['size'] * current_price for p in st.session_state.positions if p['active'] and p['margin_mode'] == 'Cross'])
        total_maint_margin = total_position_value * mmr
        
        # If Cross positions exist and Equity < Total Maint Margin (for cross positions) -> Liquidate Cross
        cross_positions = [p for p in st.session_state.positions if p['active'] and p['margin_mode'] == 'Cross']
        if cross_positions and wallet['equity'] < total_maint_margin:
             for pos in cross_positions:
                 self.liquidate(pos, current_price)
                 pos['active'] = False
                 # Actually need to capture the realized loss
        
        # Handle Isolated deletions
        for pos in st.session_state.positions:
            if pos['active']:
                if pos['margin_mode'] == 'Isolated':
                    # Re-check updated PnL logic above
                    margin_balance = pos['initial_margin'] + pos['pnl']
                    maint_margin_req = (pos['size'] * current_price) * mmr
                    if margin_balance < maint_margin_req:
                        self.liquidate(pos, current_price)
                        pos['active'] = False
                active_positions.append(pos)
                
        st.session_state.positions = active_positions

    def liquidate(self, pos, price):
        # Realize the loss (Position Margin is lost)
        # For Isolated: Loss is capped at Margin.
        # For Cross: Loss is taken from Balance.
        loss = 0
        if pos['margin_mode'] == 'Isolated':
            loss = -pos['initial_margin'] # Lose the margin
            # In update, we already removed it from balance if we finalize? No, balance is static.
            # We must subtract from Balance.
            st.session_state.wallet['balance'] += (pos['pnl'] if pos['pnl'] > -pos['initial_margin'] else -pos['initial_margin'])
            st.session_state.wallet['used_margin'] -= pos['initial_margin']
            
        else: # Cross
            st.session_state.wallet['balance'] += pos['pnl'] 
            st.session_state.wallet['used_margin'] -= pos['initial_margin']
            
        st.warning(f"Position {pos['id']} LIQUIDATED at {price:.2f}")

    def close_position(self, pos_id, current_price):
        for pos in st.session_state.positions:
            if pos['id'] == pos_id:
                # Realize PnL
                st.session_state.wallet['balance'] += pos['pnl']
                st.session_state.wallet['used_margin'] -= pos['initial_margin']
                pos['active'] = False
                st.session_state.positions.remove(pos)
                return

# ==========================================
# UI & MAIN LOOP
# ==========================================

def main():
    market = MarketEngine()
    trader = TradingEngine()
    
    # --- SIDEBAR ---
    st.sidebar.title("⚡ PriceSim")
    
    # Settings (Volatility)
    with st.sidebar.expander("⚙️ Market Settings", expanded=True):
        if 'volatility' not in st.session_state:
            st.session_state.volatility = SIGMA
            
        st.session_state.volatility = st.slider(
            "Market Volatility (Sigma)", 
            min_value=0.1, max_value=5.0, value=st.session_state.volatility, step=0.1,
            help="Higher value = more chaotic price movement"
        )
        
        col1, col2 = st.columns(2)
        if col1.button("FLASH CRASH 📉"):
             st.session_state.market_data.loc[len(st.session_state.market_data)-1, 'Close'] *= 0.95
             market.price *= 0.95
             st.toast("Flash Crash! -5%")
             
        if col2.button("PUMP IT 🚀"):
             st.session_state.market_data.loc[len(st.session_state.market_data)-1, 'Close'] *= 1.05
             market.price *= 1.05
             st.toast("Pump Triggered! +5%")

    # Wallet Info
    w = st.session_state.wallet
    st.sidebar.metric("Equity (Est)", f"${w['equity']:.2f}", delta=f"{w['pnl']:.2f}")
    st.sidebar.text(f"Wallet Balance: ${w['balance']:.2f}")
    st.sidebar.text(f"Used Margin: ${w['used_margin']:.2f}")
    st.sidebar.text(f"Available: ${w['equity'] - w['used_margin']:.2f}")
    
    st.sidebar.markdown("---")
    
    # Order Entry
    st.sidebar.subheader("Place Order")
    side = st.sidebar.radio("Side", ["Long", "Short"], horizontal=True)
    margin_mode = st.sidebar.selectbox("Margin Mode", ["Isolated", "Cross"])
    leverage = st.sidebar.slider("Leverage", 1, 50, 10)
    
    # Size Input Type
    size_type = st.sidebar.radio("Size Type", ["XYZ", "USD", "% Balance"], horizontal=True)
    
    current_price = market.price
    
    amount = 0.0
    final_size_xyz = 0.0
    
    if size_type == "XYZ":
        amount = st.sidebar.number_input("Size (XYZ)", min_value=0.001, max_value=100.0, value=0.1, step=0.01)
        final_size_xyz = amount
    elif size_type == "USD":
        amount = st.sidebar.number_input("Size (USD)", min_value=10.0, max_value=1000000.0, value=1000.0, step=100.0)
        final_size_xyz = amount / current_price
    else:
        pct = st.sidebar.slider("Use % of Available", 1, 100, 50)
        # Available for margin
        available = st.session_state.wallet['equity'] - st.session_state.wallet['used_margin']
        margin_to_use = available * (pct / 100.0)
        # Position Size = Margin * Leverage
        pos_size_usd = margin_to_use * leverage
        final_size_xyz = pos_size_usd / current_price
        
        st.sidebar.caption(f"Margin: ${margin_to_use:.2f}")
        st.sidebar.caption(f"Pos Size: ${pos_size_usd:.2f} ({final_size_xyz:.4f} XYZ)")

    if st.sidebar.button("Execute Order"):
        success, msg = trader.place_order(side, final_size_xyz, leverage, margin_mode, current_price)
        if success:
            st.sidebar.success(msg)
        else:
            st.sidebar.error(msg)
            
    # --- MAIN AREA ---
    
    # Real-time Updater
    # In Streamlit, we loop using rerun. 
    # To avoid blocking, we can use a placeholder.
    
    placeholder_chart = st.empty()
    placeholder_table = st.empty()
    
    # Run one tick of simulation per rerun or just read current state?
    # Streamlit execution model means the script reruns top to bottom.
    # We trigger a tick.
    current_price = market.tick()
    trader.update_positions(current_price)
    
    # 1. CHART
    df = st.session_state.market_data
    # Calculate Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # Create Subplots: Row 1 = Price, Row 2 = Volume
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df['Time'],
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name='Price',
        increasing_line_color= '#26a69a', decreasing_line_color= '#ef5350',
        line=dict(width=1.5) # Thicker wicks
    ), row=1, col=1)
    
    # Add SMAs
    fig.add_trace(go.Scatter(x=df['Time'], y=df['SMA_20'], mode='lines', name='SMA 20', line=dict(color='yellow', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Time'], y=df['SMA_50'], mode='lines', name='SMA 50', line=dict(color='orange', width=1.5)), row=1, col=1)
    
    # Volume Bar
    colors = ['#26a69a' if row['Close'] >= row['Open'] else '#ef5350' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df['Time'], y=df['Volume'], name='Volume', marker_color=colors), row=2, col=1)
    
    fig.update_layout(
        title=f"{SYMBOL} - ${current_price:.2f}",
        template="plotly_dark",
        height=700,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        uirevision='true' # Preserve state (zoom/pan) across reruns
    )
    
    # Add positions entry lines
    for pos in st.session_state.positions:
        color = 'green' if pos['side'] == 'Long' else 'red'
        fig.add_hline(y=pos['entry_price'], line_dash="dash", line_color=color, annotation_text=f"{pos['side']} Entry")

    placeholder_chart.plotly_chart(fig, use_container_width=True)
    
    # 2. POSITIONS TABLE
    if st.session_state.positions:
        pos_df = pd.DataFrame(st.session_state.positions)
        # Format for display
        display_cols = pos_df[['id', 'side', 'size', 'entry_price', 'leverage', 'margin_mode', 'pnl', 'liq_price']].copy()
        display_cols['entry_price'] = display_cols['entry_price'].map('${:,.2f}'.format)
        display_cols['pnl'] = display_cols['pnl'].map('${:,.2f}'.format)
        display_cols['liq_price'] = display_cols['liq_price'].map('${:,.2f}'.format)
        
        # Add Close Button (Not easy in data_editor directly without callback, so we use columns)
        st.subheader("Open Positions")
        
        for idx, row in pos_df.iterrows():
            c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
            c1.write(row['side'])
            c2.write(f"{row['size']} XYZ")
            c3.write(f"{row['entry_price']:.2f}")
            c4.write(f"{row['leverage']}x")
            c5.write(row['margin_mode'])
            c6.write(f"{row['pnl']:.2f}")
            c7.write(f"{row['liq_price']:.2f}")
            if c8.button(f"Close", key=f"close_{row['id']}"):
                trader.close_position(row['id'], current_price)
                st.rerun()
    else:
        st.info("No active positions.")

    # Auto-refresh loop
    time.sleep(1) 
    st.rerun()

if __name__ == "__main__":
    main()

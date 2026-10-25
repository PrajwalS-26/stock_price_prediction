"""Generate docs/Stock_Prediction_Project_Documentation.pdf (no network needed)."""

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, ListFlowable, ListItem,
                                NextPageTemplate, PageBreak, PageTemplate,
                                Paragraph, Preformatted, Spacer, Table, TableStyle)

OUT = "docs/Stock_Prediction_Project_Documentation.pdf"

NAVY = colors.HexColor("#0B3D66")
BLUE = colors.HexColor("#14507D")
ACCENT = colors.HexColor("#00A38A")
LIGHT = colors.HexColor("#EAF3FB")
GREYBG = colors.HexColor("#F4F5F7")
BORDER = colors.HexColor("#9EC5E8")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Normal"], fontName="Helvetica-Bold",
                    fontSize=17, leading=21, textColor=NAVY, spaceBefore=16, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=ss["Normal"], fontName="Helvetica-Bold",
                    fontSize=12.5, leading=16, textColor=BLUE, spaceBefore=12, spaceAfter=5)
BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontName="Helvetica",
                      fontSize=9.8, leading=14.2, spaceAfter=6)
BULLET = ParagraphStyle("BULLET", parent=BODY, spaceAfter=3)
CODE_S = ParagraphStyle("CODE_S", fontName="Courier", fontSize=8.4, leading=11.6)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=8.4, leading=11.5,
                       textColor=colors.HexColor("#555555"))
TITLE = ParagraphStyle("TITLE", parent=ss["Normal"], fontName="Helvetica-Bold",
                       fontSize=26, leading=31, alignment=TA_CENTER, textColor=NAVY)
SUBTITLE = ParagraphStyle("SUBTITLE", parent=ss["Normal"], fontName="Helvetica",
                          fontSize=13, leading=18, alignment=TA_CENTER, textColor=BLUE)


def p(text, style=BODY):
    return Paragraph(text, style)


def h1(text):
    return Paragraph(text, H1)


def h2(text):
    return Paragraph(text, H2)


def bullets(items, numbered=False):
    return ListFlowable(
        [ListItem(Paragraph(t, BULLET), leftIndent=14) for t in items],
        bulletType="1" if numbered else "bullet",
        bulletFontSize=8, leftIndent=16, spaceAfter=6,
    )


def code(text):
    tbl = Table([[Preformatted(text, CODE_S)]], colWidths=[16.6 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), GREYBG),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#CCCCCC")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return tbl


def table(rows, widths=None, header=True, font_size=9.0):
    data = []
    for r in rows:
        data.append([Paragraph(str(c), ParagraphStyle(
            "cell", fontName="Helvetica-Bold" if header and r is rows[0] else "Helvetica",
            fontSize=font_size, leading=font_size + 3.2)) for c in r])
    tbl = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C8D0DA")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), NAVY),
                  ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
        style += [("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT])]
    tbl.setStyle(TableStyle(style))
    return tbl


def note(title, text):
    inner = [[Paragraph(f"<b>{title}</b><br/>{text}",
                        ParagraphStyle("note", parent=BODY, fontSize=9.3, leading=13.2))]]
    tbl = Table(inner, colWidths=[16.6 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.8, BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return tbl


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(2 * cm, 1.1 * cm, "AI Stock Prediction -- Major Project Documentation")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


doc = BaseDocTemplate(OUT, pagesize=A4,
                      leftMargin=2 * cm, rightMargin=2 * cm,
                      topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                      title="AI Stock Prediction - Project Documentation")
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=on_page)])

S = []  # story

# ============================== TITLE PAGE ==============================
S += [Spacer(1, 2.2 * cm),
      p("AI STOCK MARKET PREDICTION", TITLE),
      Spacer(1, 0.3 * cm),
      p("NSE Equity Price Forecasting, Trading Signals,<br/>News Sentiment &amp; Walk-Forward Backtesting", SUBTITLE),
      Spacer(1, 1.2 * cm),
      table([
          ["Item", "Detail"],
          ["Problem", "Predict tomorrow's closing price of any stock and derive a BUY / HOLD / SELL signal"],
          ["Models", "XGBoost (x4) + LSTM + GRU + Inverse-RMSE weighted Ensemble"],
          ["Data", "Yahoo Finance via yfinance (prices) + RSS news (sentiment)"],
          ["Sentiment", "FinBERT transformer (offline-cached) with VADER fallback"],
          ["Validation", "Chronological splits + walk-forward backtest (no look-ahead)"],
          ["Interface", "Single Streamlit web app (dark theme) with 5 pages"],
          ["Stack", "Python 3.11, pandas, numpy, scikit-learn, XGBoost, TensorFlow/Keras, Streamlit, Plotly"],
      ], widths=[3.4 * cm, 13.2 * cm]),
      Spacer(1, 1.0 * cm),
      note("Academic disclaimer",
           "This is a college major project. It is NOT investment advice. Stock markets are "
           "inherently unpredictable; a model that works historically is not guaranteed to work "
           "in the future. Never invest real money based only on this tool."),
      Spacer(1, 0.6 * cm),
      p("This document explains the complete working of the project for a reader with no prior "
        "knowledge: every page of the app, every model, every calculation, and exactly how the "
        "news sentiment engine works.", SUBTITLE),
      PageBreak()]

# ============================== 1. OVERVIEW ==============================
S += [h1("1. What This Project Does"),
      p("You type the stock symbol of any company (for example <b>TCS.NS</b> for Tata Consultancy "
        "Services on the Indian NSE exchange). The app then:"),
      bullets([
          "<b>downloads</b> that company's full daily price history from Yahoo Finance (up to 10 years),",
          "<b>builds features</b> -- numerical summaries of price behaviour (averages, momentum, volatility, etc.),",
          "<b>trains</b> several machine-learning models on the spot (about 50 seconds the first time for a company),",
          "<b>predicts</b> tomorrow's return and converts it into a <b>BUY / HOLD / SELL</b> signal,",
          "<b>reads the news</b> headlines about the company with a financial-language AI (FinBERT) and lets that "
          "sentiment softly adjust the signal,",
          "<b>proves itself</b> with a walk-forward backtest -- a simulation of what would have happened in the past "
          "if the signal had been traded, including trading costs, compared against simply buying and holding.",
      ]),
      p("Nothing is pre-trained for a fixed list of companies. Training happens <b>on demand</b>, "
        "only for the ticker you ask for, and the result is cached in the "
        "<font face='Courier'>saved_models/</font> folder so revisits are instant."),
      h2("Why predict a return and not a price directly?"),
      p("A share price can be Rs.50 or Rs.50,000 -- raw price values are impossible to compare across "
        "companies and drift endlessly upward. Models here predict the <b>percentage return for the next day</b>:"),
      code("return(t)  =  price(t+1) / price(t)  -  1        # e.g. +0.012 = +1.2% tomorrow"),
      p("The predicted price is then reconstructed as "
        "<font face='Courier'>predicted_price = current_price x (1 + predicted_return)</font>. "
        "This one design decision is what kept the old project's error (MAPE) at a realistic ~1.1% "
        "instead of 8%+.")]

# ============================== 2. PIPELINE ==============================
S += [h1("2. How It Works at a Glance"),
      p("The whole system is a six-step pipeline. Each step is explained in detail later:"),
      table([
          ["Step", "What happens", "Code module"],
          ["1. Data", "Download OHLCV prices for the ticker + NIFTY 50 + India VIX; fetch news headlines via RSS",
           "data/data_loader.py, data/news_loader.py"],
          ["2. Features", "Turn raw prices into 16 numerical features + forward-return targets",
           "features/feature_engineering.py"],
          ["3. Models", "Train XGBoost (x4), LSTM, GRU; combine them with inverse-RMSE weights",
           "models/train.py, models/sequence.py, models/ensemble.py"],
          ["4. Signal", "XGBoost 1-day return vs +/-0.5% threshold -> BUY/HOLD/SELL, tilted by news sentiment",
           "models/predict.py"],
          ["5. Backtest", "Walk-forward simulation: retrain every 21 trading days, pay 10 bps per trade, "
           "compare vs buy &amp; hold", "models/backtest.py"],
          ["6. Log", "Store each prediction in SQLite; score direction accuracy once actuals arrive",
           "utils/prediction_log.py"],
      ], widths=[2.2 * cm, 9.6 * cm, 4.8 * cm]),
      Spacer(1, 0.2 * cm),
      code("ticker -> yfinance data -> features -> [XGBoost 1d/5d/21d + direction] [LSTM] [GRU]\n"
           "                                      \\___________ inverse-RMSE ensemble _______/\n"
           "                                                |\n"
           "                 news RSS -> FinBERT score -----+--> BUY / HOLD / SELL signal\n"
           "                                                |\n"
           "                              walk-forward backtest + prediction log")]

# ============================== 3. BEGINNER CONCEPTS ==============================
S += [h1("3. Basic Concepts (for First-Time Readers)"),
      table([
          ["Term", "Meaning"],
          ["Ticker", "A symbol Yahoo Finance uses to identify a company. Indian NSE stocks end in <b>.NS</b> "
           "(TCS.NS, INFY.NS, RELIANCE.NS). Indices use <b>^</b> ( ^NSEI = NIFTY 50, ^INDIAVIX = India VIX ). "
           "US tickers work too (AAPL)."],
          ["OHLCV", "For every trading day: Open, High, Low, Close prices and Volume (shares traded). "
           "The chart page draws these as candlesticks."],
          ["Return", "Percent change between two closes: <font face='Courier'>close(t)/close(t-1) - 1</font>. "
           "Returns, not prices, are what the models learn."],
          ["Train / Test", "Models learn from older data (train) and are scored on newer data they have never "
           "seen (test). Our split is strictly chronological -- the last 15% of history is the test set. "
           "No shuffling, because time order matters."],
          ["Overfitting", "When a model memorises past noise instead of learning patterns; it scores great on "
           "training data but fails on new data. Guarded against by chronological testing, walk-forward "
           "validation, and comparing against a naive baseline."],
          ["Look-ahead bias", "The worst sin in finance modelling: accidentally letting the model see the "
           "future while training. Prevented everywhere: scalers fit on training data only, features use only "
           "past values, backtest retrain steps use only data available on that day."],
          ["Naive baseline", "The deliberately dumb predictor 'tomorrow's price = today's price'. Every metric "
           "is reported next to this baseline; a model that cannot beat it has no real skill."],
      ], widths=[3.2 * cm, 13.4 * cm])]

# ============================== 4. DATA ==============================
S += [h1("4. Step 1 -- Where the Data Comes From"),
      h2("Price data: yfinance"),
      p("<font face='Courier'>yfinance</font> is a free, unofficial Python library that reads Yahoo Finance. "
        "One call returns a table of daily OHLCV for the requested ticker and history length "
        "(1y / 2y / 5y / 10y). The adjusted close (price corrected for splits and dividends) is used as "
        "<b>Price</b>."),
      p("Three series are downloaded: the <b>stock itself</b>, <b>NIFTY 50 (^NSEI)</b> as the market "
        "benchmark, and <b>India VIX (^INDIAVIX)</b> as a fear gauge. All downloads are cached as CSV "
        "files under <font face='Courier'>data/cache/</font> for 24 hours, so repeated page loads do not "
        "hammer Yahoo and the app still renders when a fetch hiccupes."),
      h2("News data: RSS feeds (no API key needed)"),
      p("Two public RSS feeds are read with plain HTTP + XML parsing (no paid API):"),
      bullets([
          "Yahoo Finance RSS, per ticker: <font face='Courier'>feeds.finance.yahoo.com/...headline?s=TCS.NS...</font>",
          "Google News RSS, searched by the company's real name (resolved once from Yahoo), which finds "
          "local/business articles RSS might miss.",
      ]),
      p("Headlines are de-duplicated and limited (typically 30-75 recent items). This is the raw material "
        "for the sentiment engine described in Section 8.")]

# ============================== 5. FEATURES ==============================
S += [h1("5. Step 2 -- Feature Engineering"),
      p("Models do not see prices; they see <b>features</b> -- 16 numbers per day that summarise the "
        "recent state of the market. Think of them as the model's instruments and gauges."),
      table([
          ["Feature", "How it is calculated", "Plain-English meaning"],
          ["Return", "price(t)/price(t-1) - 1", "Today's percentage move"],
          ["MA10", "10-day rolling average of price", "Short-term trend line"],
          ["MA20", "20-day rolling average of price", "Medium-term trend line"],
          ["Volatility", "10-day rolling std-dev of daily returns", "How jumpy the stock is right now"],
          ["RSI", "14-day Relative Strength Index (0-100)", "Overbought (&gt;70) vs oversold (&lt;30) momentum"],
          ["MACD", "EMA12 - EMA26 of price", "Trend-following momentum line"],
          ["MACD_signal", "9-day EMA of the MACD line", "Crossover trigger for MACD"],
          ["BB_high / BB_low", "20-day mean +/- 2 standard deviations", "Bollinger volatility bands -- price position in range"],
          ["Momentum", "price(t) - price(t-10)", "Rupee change over the last 10 days"],
          ["Lag1 / Lag2 / Lag3", "returns of 1, 2 and 3 days ago", "Recent memory of daily moves"],
          ["Excess_Return", "stock return - NIFTY return same day", "Did it beat the market today?"],
          ["Mkt_Vol", "10-day rolling std of NIFTY returns", "Is the whole market nervous?"],
          ["VIX", "India VIX index level (forward-filled)", "Market-wide fear gauge (high = fearful)"],
      ], widths=[3.0 * cm, 6.0 * cm, 7.6 * cm]),
      Spacer(1, 0.2 * cm),
      h2("Targets -- what the models learn to predict"),
      code("Target      = price(t+1) / price(t)  - 1      # next-day return (main)\n"
           "Target_5d   = price(t+5) / price(t)  - 1      # one-week return\n"
           "Target_21d  = price(t+21) / price(t) - 1      # one-month return"),
      p("The last rows have no target yet (the future has not happened) -- they are dropped from "
        "training. The final row's feature vector is exactly what is fed to the trained model to "
        "forecast tomorrow."),
      note("Why add NIFTY and VIX features at all?",
           "A stock's future depends on its own history AND the market mood. Excess_Return, Mkt_Vol and "
           "VIX let the model learn regime patterns -- for example 'high-VIX days tend to mean-revert' -- "
           "which is also how we compensate for not having years of historical news (see Section 8).")]

# ============================== 6. MODELS ==============================
S += [h1("6. Step 3 -- The Machine-Learning Models"),
      p("Four model families are trained for each ticker. All outputs are <b>returns</b>; prices are "
        "reconstructed afterwards."),
      h2("6.1 XGBoost (gradient-boosted decision trees)"),
      p("XGBoost builds many small decision trees in sequence; each new tree focuses on the mistakes of "
        "the trees before it. It is the champion for tabular (spreadsheet-like) data. We train four of them:"),
      table([
          ["Model", "Predicts", "Used for"],
          ["xgb_1d", "next-day return", "<b>the trading signal</b> + display target"],
          ["xgb_5d", "5-day return", "'1 week expected %' table on the Overview page"],
          ["xgb_21d", "21-day return", "'1 month expected %' table"],
          ["xgb_direction", "class UP (1) / DOWN (0)", "direction classifier metric on Model Performance page"],
      ], widths=[3.2 * cm, 5.2 * cm, 8.2 * cm]),
      h2("6.2 LSTM (long short-term memory) -- original sequence model"),
      p("A recurrent neural network that reads a <b>window of the last 30 days of returns</b>, one day "
        "at a time, keeping a memory of what it has seen, and outputs a single number: the predicted "
        "return for day 31. Its MinMax scaler is fit on the training window only (no look-ahead)."),
      h2("6.3 GRU (gated recurrent unit) -- newly added"),
      p("GRU is a newer, streamlined sibling of the LSTM. Where the LSTM uses three gates plus a separate "
        "cell-state pipeline, the GRU compresses this into just <b>two gates -- reset and update</b>:"),
      bullets([
          "<b>Reset gate</b>: decides how much of the old memory to forget when reading the new day's return.",
          "<b>Update gate</b>: decides how much of the old memory to carry forward vs replace.",
      ]),
      p("Fewer gates mean fewer parameters, so a GRU trains faster and needs less data than an LSTM while "
        "usually matching its accuracy on short windows like ours. It consumes the exact same 30-day "
        "scaled-return sequence as the LSTM and also emits one next-day return. In the trained TCS model "
        "it even ended up with the largest ensemble weight (0.37), a nice payoff for adding it."),
      h2("6.4 Inverse-RMSE weighted ensemble -- newly added"),
      p("Three price predictions are better than one -- if they are weighted by trust. After training, "
        "each model is scored on the held-out validation window; the weight of each model is the "
        "inverse of its error, normalised so the weights sum to 1:"),
      code("weight(i)  = (1 / RMSE_i) / sum( 1 / RMSE_j  for all models j )\n"
           "\n"
           "example:  xgb RMSE = 40, lstm RMSE = 30, gru RMSE = 29  (rupees)\n"
           "  inv     = 0.0250 , 0.0333 , 0.0345\n"
           "  weights = 0.27   , 0.36   , 0.37      (they sum to 1.0)\n"
           "\n"
           "final_price = 0.27*xgb_price + 0.36*lstm_price + 0.37*gru_price"),
      p("A lower-RMSE (more accurate) model automatically gets a bigger say. If any model fails to "
        "produce a number, it is dropped and the remaining weights are re-scaled."),
      note("Important design note -- what drives the BUY/SELL signal?",
           "The signal comes ONLY from the XGBoost 1-day model -- the same model validated in the "
           "walk-forward backtest. Sequence models (LSTM/GRU) extrapolate momentum and can be "
           "over-optimistic, so their outputs are shown as reference target prices, not as the trade "
           "trigger. The ensemble price is likewise a display target.")]

# ============================== 7. SIGNAL ==============================
S += [h1("7. Step 4 -- From Prediction to Trading Signal"),
      p("The XGBoost 1-day expected return is compared against a +/-0.5% dead-zone threshold "
        "(0.005 as a fraction; this value was chosen from the backtest):"),
      table([
          ["Predicted 1-day return", "Model signal"],
          ["greater than +0.5%", "BUY"],
          ["between -0.5% and +0.5%", "HOLD"],
          ["less than -0.5%", "SELL"],
      ], widths=[8.3 * cm, 8.3 * cm]),
      p("Example: current price Rs.2,000 and the model predicts +0.8% (Rs.2,016) -- that is above the "
        "threshold, so the base signal is BUY. A prediction of +0.2% is inside the dead zone: HOLD."),
      h2("Sentiment tilt &amp; Multi-Factor Consensus"),
      p("The base signal is then integrated with the news score (calibrated for 50-75 headlines). Rules:"),
      table([
          ["News score", "Model signal", "Final signal", "Risk Management &amp; Confidence"],
          ["&gt;= +0.05 (positive)", "BUY", "<b>BUY</b>", "High (Technicals &amp; News agree on upside)"],
          ["&lt;= -0.03 (negative)", "BUY", "<b>HOLD</b> (hedged)", "Low (Conflict: never buy into negative news)"],
          ["&gt;= +0.05 (positive)", "SELL", "<b>HOLD</b> (softened)", "Low (Conflict: positive news may trigger rally)"],
          ["&lt;= -0.03 (negative)", "SELL", "<b>SELL</b>", "High (Technicals &amp; News agree on downside)"],
          ["&gt;= +0.05 (positive)", "HOLD", "<b>BUY</b>", "High (Positive news catalyst upgrades neutral)"],
          ["&lt;= -0.03 (negative)", "HOLD", "<b>SELL</b>", "High (Negative news breaks down neutral)"],
          ["between -0.03 and +0.05", "any", "unchanged", "Medium (Neutral news preserves technical signal)"],
      ], widths=[4.2 * cm, 2.8 * cm, 3.4 * cm, 6.2 * cm]),
      p("Risk Management Rule: If the technical model predicts price growth but news sentiment is "
        "negative (e.g. -4.8/100 on TCS), buying is dangerous. The system hedges by downgrading "
        "the signal to HOLD to avoid buying into surprise downside headlines.")]

# ============================== 8. SENTIMENT ==============================
S += [h1("8. Sentiment Analysis -- Complete Deep Dive"),
      h2("8.1 Why read the news at all?"),
      p("Prices react to information: an earnings blowout, a regulator fine, a CEO resignation. The "
        "price history alone cannot see tomorrow's headline, but reading today's headlines tells us the "
        "mood around a company. The app measures that mood as a single number (score) in [-1, +1]."),
      h2("8.2 Stage 1 -- collect headlines (RSS)"),
      p("As described in Section 4: Yahoo Finance RSS for the exact ticker + Google News RSS for the "
        "company name. Typical yield: 30-75 recent headlines with title, date and source. No API key, "
        "no cost."),
      h2("8.3 Stage 2 -- score each headline with FinBERT"),
      p("<b>FinBERT</b> is a BERT language model -- a neural network pre-trained on huge amounts of "
        "English text, then further fine-tuned by researchers (ProsusAI) on financial writing. Because "
        "it is a transformer, it understands context and word order, which matters a lot in finance: "
        "it knows that 'profits fell' and 'losses fell' are opposite sentiments even though both "
        "contain similar words."),
      p("For each headline it outputs three probabilities -- P(positive), P(neutral), P(negative) -- "
        "which are turned into one score:"),
      code("headline_score = P(positive) - P(negative)        # lies in [-1, +1]\n"
           "\n"
           "example  'TCS Q4 profit jumps 40%, beats estimates'\n"
           "         -> P(pos)=0.94 P(neu)=0.05 P(neg)=0.01  -> score = +0.93\n"
           "\n"
           "example  'TCS shares fall as IT spending worries grow'\n"
           "         -> P(pos)=0.02 P(neu)=0.11 P(neg)=0.87  -> score = -0.85"),
      p("The model (about 400 MB) is downloaded <b>once</b> from Hugging Face and cached on disk; after "
        "that it runs fully <b>offline</b> (the code forces offline mode), so scoring adds no network "
        "delay and cannot hang on connection issues. Results are additionally cached in memory for "
        "15 minutes per ticker."),
      h2("8.4 Fallback -- VADER"),
      p("If FinBERT cannot load (fresh machine, no cache), the app falls back to <b>VADER</b>, a "
        "classical rule-based sentiment tool: it scores words from a hand-built financial lexicon and "
        "applies grammar rules (negation like 'not good', intensifiers like 'very', capitalisation, "
        "exclamation). No AI weights -- pure rules -- so it is instant and dependency-free, though less "
        "accurate than FinBERT."),
      h2("8.5 Stage 3 -- aggregate to one score per company"),
      code("company_score = mean( headline_score for every headline fetched )\n"
           "\n"
           "Aggregated thresholds (calibrated for 40-75 headlines):\n"
           "  score >= +0.05  ->  'Positive' (Sentiment Verdict: BUY)\n"
           "  score <= -0.03  ->  'Negative' (Sentiment Verdict: SELL)\n"
           "  else            ->  'Neutral'  (Sentiment Verdict: HOLD)"),
      p("When 40 to 75 headlines are averaged, individual positive and negative headlines balance out, "
        "so the average rarely exceeds +/-0.20. Calibrated thresholds (+0.05 and -0.03) ensure realistic "
        "and statistically meaningful sentiment signals (e.g., a score of -4.8/100 on TCS accurately reflects "
        "predominantly bearish news and yields a SELL sentiment verdict). This feeds the multi-factor decision matrix."),
      note("Honest limitation (stated in the UI too)",
           "Free RSS feeds only expose RECENT news, not a 10-year archive, so historical sentiment "
           "cannot be a training feature -- the model cannot learn 'bad news days crash' from data it "
           "never saw. Instead: (a) the trained models learn market mood from the NIFTY/VIX regime "
           "features, and (b) live news softly tilts the live signal, exactly as described.")]

# ============================== 9. BACKTEST ==============================
S += [h1("9. Step 5 -- Walk-Forward Backtest"),
      h2("9.1 Why not just test on the last 15%?"),
      p("That tests one fixed model. In reality you would retrain periodically as new data arrives. A "
        "<b>walk-forward</b> backtest mimics that real workflow and, crucially, can never peek at the "
        "future."),
      h2("9.2 The algorithm (as implemented)"),
      bullets([
          "Take the full feature frame; drop rows with no target yet.",
          "First <b>50%</b> of the days = initial training window; the remaining days are traded in the simulation.",
          "Walk forward one day at a time. Every <b>21 trading days</b> (and at the start), refit a fresh "
          "XGBoost on <b>all data strictly before today</b> -- an expanding window, never including today or later.",
          "Each day: predict tomorrow's return. Predicted return &gt; +0.5% -> <b>position = 1 (long)</b>, "
          "otherwise <b>position = 0 (flat, sit in cash)</b>. No shorting.",
          "Apply reality: every time the position flips (0 to 1 or 1 to 0), pay <b>10 bps (0.1%)</b> in costs.",
          "Daily strategy return = position x that day's actual return - turnover x cost.",
          "Compound everything into an equity curve and compare with buying the stock on day one and "
          "holding (buy &amp; hold), starting from Rs.10,000.",
      ], numbered=True),
      h2("9.3 The metrics shown on the Backtest page"),
      table([
          ["Metric", "Definition"],
          ["Total return %", "Final value / initial - 1, in percent"],
          ["CAGR %", "Compound annual growth rate: (final/initial)^(1/years) - 1"],
          ["Sharpe ratio", "mean(daily return) / std(daily return) x sqrt(252) -- return per unit of risk; higher is better"],
          ["Max drawdown %", "Worst peak-to-trough fall of the equity curve, in percent; the pain you would have felt"],
          ["Trades taken", "How many times the position flipped (each pays 10 bps)"],
          ["Win rate %", "Share of days the strategy was in the market and the day's return was positive"],
          ["Directional accuracy %", "Share of days the model predicted the correct up/down direction"],
      ], widths=[4.0 * cm, 12.6 * cm]),
      p("A strategy that halves the drawdown of buy &amp; hold while giving up some upside is doing "
        "exactly its job: risk control. (Real TCS example in Section 13.)")]

# ============================== 10. LOG ==============================
S += [h1("10. Step 6 -- Prediction Logging &amp; Accuracy Tracking"),
      p("Every time the dashboard opens a ticker (once per browser session), one row is written to a "
        "SQLite database (<font face='Courier'>data/prediction_log.db</font>):"),
      code("ticker | predicted_on (next trading day) | current_price | predicted_price (XGB)\n"
           "       | signal | model_signal | sentiment_score"),
      p("When you open the app again on a later day, the actual close for the forecasted day is known, "
        "so the log resolves the row: was the predicted direction right or wrong? The Signal Log page "
        "shows predictions logged, how many have been resolved, and the running <b>direction accuracy</b> "
        "-- the project's own live scorecard, which needs no manual work and grows forever.")]

# ============================== 11. UI WALKTHROUGH ==============================
S += [h1("11. The App, Page by Page"),
      p("Run <font face='Courier'>streamlit run app.py</font> and open http://localhost:8501. "
        "The app has a permanent sidebar plus five tabs."),
      h2("Sidebar (visible on every page)"),
      bullets([
          "<b>Stock ticker</b> -- any Yahoo symbol (TCS.NS, INFY.NS, RELIANCE.NS, ^NSEI, AAPL ...).",
          "<b>Training history</b> -- 10y / 5y / 2y / 1y of data to train on.",
          "<b>Load &amp; Predict button</b> -- fetches + trains + shows the dashboard automatically. "
          "First time for a ticker shows a live progress bar (data fetch -> XGBoost 1d/5d/21d -> "
          "direction -> LSTM -> GRU -> ensemble -> save), about 50 seconds. Later visits are instant "
          "(cached model).",
          "<b>Model status card</b> -- when it was trained, on how many rows, over which date range.",
          "<b>Retrain button</b> -- forces a fresh training run (e.g. next day, to fold in new data).",
          "A bad ticker or dead connection produces a clear error with a 'Try again' button -- never a crash.",
      ]),
      PageBreak(),
      h2("Page 1 -- Overview"),
      bullets([
          "<b>Hero banner</b>: ticker, full company name (from Yahoo), current price and today's change "
          "% (last two closes).",
          "<b>Four target cards</b>: XGBoost / LSTM / GRU targets, each "
          "<font face='Courier'>current_price x (1 + that model's predicted return)</font>, plus the "
          "<b>ensemble target</b> = the inverse-RMSE weighted blend, with its expected return %.",
          "<b>Trading signal badge</b> (big coloured BUY / HOLD / SELL pill): the XGBoost 1-day return "
          "vs the +/-0.5% threshold, after the sentiment tilt. Under it: the model's raw signal, the "
          "sentiment label + score, and the direction classifier's UP/DOWN call.",
          "<b>Expected returns table</b>: 1-day, 5-day and 21-day return forecasts from the three "
          "XGBoost horizon models.",
          "<b>Price chart</b>: one year of candlesticks (each candle = one day's open-high-low-close) "
          "with MA10 (teal) and MA20 (gold) trend lines, and a volume histogram underneath.",
      ]),
      h2("Page 2 -- Model Performance"),
      bullets([
          "<b>Test-set evaluation table</b> (last 15% of history, chronological): for XGBoost, LSTM, GRU "
          "and the Ensemble - RMSE in rupees, the naive baseline's RMSE, MAPE %, directional accuracy %, "
          "<b>skill = 1 - model_RMSE / naive_RMSE</b> (positive means it beats 'tomorrow = today'), and R-squared.",
          "<b>Multi-horizon table</b>: the 5-day and 21-day XGBoost models' validation metrics.",
          "<b>Direction classifier card</b>: accuracy of the UP/DOWN classifier next to the majority-class "
          "base rate -- the number to beat.",
          "<b>Ensemble weights chart</b>: the inverse-RMSE weights actually saved with the model.",
          "<b>Actual vs Predicted chart</b>: the test window with the real price (solid) and every model's "
          "forecast (dotted, ensemble in red).",
          "<b>Feature importance</b>: XGBoost 'gain' -- which of the 16 features mattered most to the trees, "
          "as a horizontal bar chart.",
      ]),
      h2("Page 3 -- Backtest"),
      bullets([
          "Header metrics: strategy total return vs buy &amp; hold, strategy max drawdown, Sharpe.",
          "Full comparison table: return, CAGR, Sharpe, max drawdown, final value of Rs.10,000.",
          "Trades taken, win rate (days in market), directional accuracy.",
          "<b>Equity curve</b>: two lines from Rs.10,000 -- the strategy and buy &amp; hold -- over the "
          "test window. Everything computed by the walk-forward engine of Section 9 (refit every 21 days, "
          "10 bps costs, no look-ahead).",
      ]),
      h2("Page 4 -- Sentiment"),
      bullets([
          "<b>Gauge</b>: company score x 100, from -100 (very negative) to +100 (very positive), with the "
          "text label (threshold +/-0.3).",
          "<b>Headlines analysed</b> count and the engine used (finbert or vader fallback).",
          "<b>Impact box</b>: exactly how the tilt was applied -- 'model says X -> sentiment-adjusted signal "
          "is Y (confidence Z)'.",
          "<b>Headline list</b> with publication date and source, so you can verify the scores yourself.",
      ]),
      h2("Page 5 -- Signal Log"),
      bullets([
          "Three summary cards: predictions logged, resolved (actual price now known), running direction "
          "accuracy %.",
          "Table of the last 30 logged predictions with their outcome -- the project's self-scoring "
          "scorecard (Section 10).",
      ])]

# ============================== 12. RUN ==============================
S += [h1("12. How to Run the Project"),
      code("# 1. one-time setup\n"
           "python -m venv .venv\n"
           "source .venv/Scripts/activate      # Windows Git Bash\n"
           "pip install -r requirements.txt\n\n"
           "# 2. start the app\n"
           "streamlit run app.py               # open http://localhost:8501\n\n"
           "# optional: train tickers from the command line (multiple supported)\n"
           "python train.py TCS.NS INFY.NS --period 10y\n\n"
           "# tests\n"
           "python -m pytest                   # 42 offline tests\n"
           "python -m pytest -m network        # + 2 live end-to-end tests"),
      p("In VS Code: select the .venv interpreter (Ctrl+Shift+P -> Python: Select Interpreter), then "
        "Run &amp; Debug -> 'Streamlit app'.")]

# ============================== 13. RESULTS ==============================
S += [h1("13. Example Results &amp; Honest Limitations"),
      h2("Results (TCS.NS, 10 years of data, 5-year test window)"),
      bullets([
          "Forecast error: MAPE about 1.1% (naive-baseline-comparable at 1-day horizon -- daily noise "
          "dominates; this matches the well-known Efficient Market Hypothesis and is stated honestly).",
          "Ensemble weights: xgb 0.27 / lstm 0.36 / gru 0.37.",
          "Walk-forward backtest: the long/flat strategy roughly <b>halved the maximum drawdown</b> "
          "versus buy &amp; hold by sitting in cash during the worst stretches, at the cost of some "
          "upside -- demonstrated risk control rather than guaranteed alpha.",
          "INFY.NS and WIPRO.NS trained on demand as further proofs of the any-ticker flow.",
      ]),
      h2("Limitations"),
      bullets([
          "One stock's daily history is a small, noisy dataset -- predictions are indicative, never certain.",
          "No historical news archive -> sentiment tilts only the live signal.",
          "10 bps cost and long/flat positions are simplifications (no intraday fills, slippage, taxes).",
          "Retraining is manual (button) or per page-load of a new ticker; there is no scheduled retrain.",
      ]),
      h2("Future scope ideas"),
      bullets([
          "Sector/macro features and pair data.",
          "Transformer-based price sequence models.",
          "Optuna hyperparameter tuning; confidence-interval predictions.",
          "Scheduled daily retraining and alerting; paper-trading loop.",
      ])]

# ============================== 14. FILES ==============================
S += [h1("14. Project File Map"),
      table([
          ["Path", "Role"],
          ["app.py", "The entire Streamlit UI (sidebar auto-train flow + 5 pages)"],
          ["train.py", "Command-line training entry point"],
          ["data/data_loader.py", "yfinance downloads + 24h CSV cache"],
          ["data/news_loader.py", "RSS headline fetching + company-name resolution"],
          ["features/feature_engineering.py", "16 features + multi-horizon targets"],
          ["features/pipeline.py", "Builds the full feature frame (stock + NIFTY + VIX)"],
          ["features/sentiment.py", "FinBERT scoring (offline) + VADER fallback + caching"],
          ["features/splits.py", "Chronological and walk-forward split generators"],
          ["models/train.py", "Trains every model, computes ensemble weights, saves the bundle"],
          ["models/sequence.py", "LSTM/GRU training (train-only scaler, 30-day windows)"],
          ["models/ensemble.py", "Inverse-RMSE weights + weighted price blend"],
          ["models/predict.py", "Next-day inference, signal thresholds, sentiment tilt rules"],
          ["models/backtest.py", "Walk-forward simulation engine + metrics"],
          ["models/metrics.py", "RMSE / MAPE / directional accuracy / skill vs naive"],
          ["models/registry.py", "Per-ticker model bundles under saved_models/&lt;ticker&gt;/"],
          ["models/explain.py", "XGBoost feature importances"],
          ["utils/prediction_log.py", "SQLite predictions + accuracy resolution"],
          ["tests/", "42 offline unit tests + 2 live UI end-to-end tests"],
          [".streamlit/config.toml", "Dark theme + server settings"],
      ], widths=[6.0 * cm, 10.6 * cm]),
      Spacer(1, 0.4 * cm),
      note("Congratulations",
           "If you have read this far, you can explain every screen, every model and every number in "
           "the app -- including exactly how news headlines become a graded trading signal. "
           "Good luck with the project demo!")]


# ============================== 15. VIVA QUESTIONS ==============================
S += [h1("15. Major Project Viva &amp; Presentation Q&amp;A"),
      p("Top questions examiners ask during project evaluation and the exact technical answers:"),
      h2("Q1: Why predict percentage returns instead of raw closing prices?"),
      p("<b>Answer:</b> Raw prices are non-stationary; over 5-10 years they drift into ranges the model has "
        "never seen before, causing severe extrapolation errors. Percentage returns fluctuate around a "
        "stationary mean of zero, enabling statistical models to generalize accurately across different price regimes."),
      h2("Q2: How did you prevent data leakage in sequence modeling?"),
      p("<b>Answer:</b> Time-series data must never be shuffled. We used strict chronological splitting "
        "(70% train, 15% validation, 15% test). Furthermore, feature scalers (MinMaxScaler) were fitted "
        "exclusively on the training window and only applied to transform the test set, ensuring zero future "
        "information leaked into the past."),
      h2("Q3: Why combine XGBoost with LSTM and GRU in an ensemble?"),
      p("<b>Answer:</b> XGBoost excels at finding complex non-linear decision boundaries across tabular technical "
        "indicators (RSI, Bollinger Bands, Volatility). However, it does not naturally maintain sequential state. "
        "LSTMs and GRUs treat prices as an evolving 30-day temporal sequence. Combining them through inverse-RMSE "
        "weighting creates an ensemble that captures both indicator relationships and time-series momentum."),
      h2("Q4: Why doesn't negative sentiment always flip the signal to SELL?"),
      p("<b>Answer:</b> In quantitative multi-factor systems, news sentiment acts as a risk hedge rather than a "
        "dominant dictator. If technical indicators are bullish but news is negative, immediately shorting the stock "
        "is risky. Instead, the system hedges by downgrading the signal to HOLD, protecting capital by refusing to "
        "buy into negative headlines while avoiding premature short trades."),
      h2("Q5: What is the primary advantage demonstrated by your backtest?"),
      p("<b>Answer:</b> Financial markets have high entropy; generating pure alpha on daily stock data is difficult "
        "due to the Efficient Market Hypothesis. However, our walk-forward backtest demonstrates a significant reduction "
        "in Maximum Drawdown compared to Buy &amp; Hold. By stepping into cash (HOLD / Flat) during bearish regimes, "
        "the strategy preserves capital and manages downside risk effectively.")]

doc.build(S)
print("PDF written:", OUT)

import yfinance as yf
import argparse

def download(ticker, start, end, out):
    df = yf.download(ticker, start=start, end=end)
    df.to_csv(out)
    print(f"Saved {len(df)} rows to {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    
    download(args.ticker, args.start, args.end, args.out)

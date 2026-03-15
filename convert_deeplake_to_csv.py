import deeplake
import pandas as pd

# Authentication token
token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiNTNhNDYyYWYtMmRjZS00Zjk4LWI3NTQtZWQ3NjEyN2I0OTU1Iiwib3JnX2lkIjoiNmFiMzUzMjQtOTEyZS00Mjc2LWEzMDktMTU5YTk3YzRiNTU3IiwidHlwZSI6ImFwaV90b2tlbiIsIm5hbWUiOiJmaXJzdF90ZXN0X0cxIiwiZXhwIjoxODA1MDY4ODAwLCJpYXQiOjE3NzM1NTM0OTZ9.FwNRueK1SannTUy9J69BvD2ZIbeGhKDg9jmdN21NgLU'

# Try different URL formats
url_formats = [
    "hub://siyuliu4262s-organization/lightwheel_bevorg_frames",
    "hub://siyuliu4262s/lightwheel_bevorg_frames",
    "siyuliu4262s-organization/lightwheel_bevorg_frames",
    "siyuliu4262s/lightwheel_bevorg_frames"
]

print("Trying to load Deep Lake dataset...")

ds = None
for url in url_formats:
    try:
        print(f"Attempting: {url}")
        ds = deeplake.load(url, token=token)
        print(f"✅ Successfully loaded from: {url}")
        break
    except Exception as e:
        print(f"❌ Failed: {e}")
        continue

if ds is None:
    print("❌ Could not load dataset with any URL format")
    print("Please check the organization name and dataset name")
    exit(1)

print(f"Dataset loaded successfully!")
print(f"Dataset length: {len(ds)}")

# Convert to dataframe
print("Converting to pandas DataFrame...")
df = ds.pandas()

print(f"DataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)[:10]}...")

# Save CSV
print("Saving to CSV...")
df.to_csv("lightwheel_bevorg_frames.csv", index=False)

print("✅ Saved CSV successfully to lightwheel_bevorg_frames.csv")
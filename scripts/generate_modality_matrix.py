"""
generate_modality_matrix.py
RM-VMusic Phase 7B: Multimodal Physical Availability Matrix Generator.

Outputs:
- data/processed/final12_modality_matrix.csv
"""

import sys
import os
import pandas as pd
from pathlib import Path

# UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_CSV = BASE_DIR / "data" / "processed" / "final_12class_metadata.csv"
OUTPUT_CSV = BASE_DIR / "data" / "processed" / "final12_modality_matrix.csv"

def generate_modality_matrix():
    print("=== RM-VMusic Phase 7B: Generating Multimodal Physical Availability Matrix ===")
    
    df = pd.read_csv(DATASET_CSV)
    
    matrix_rows = []
    for idx, row in df.iterrows():
        sid = row["song_id"]
        genre = row["genre"]
        artist = row["artist"]
        
        has_a = str(row["audio_status"]) == "verified_local"
        has_l = str(row["lyrics_status"]) == "verified_local"
        has_c = str(row["cover_status"]) == "verified_local"
        has_y = str(row["year_status"]) == "verified"
        
        matrix_rows.append({
            "song_id": sid,
            "genre": genre,
            "artist": artist,
            "audio_available": has_a,
            "lyrics_available": has_l,
            "cover_available": has_c,
            "release_year_verified": has_y,
            "audio_path": row.get("audio_path", "") if has_a else "",
            "lyrics_path": row.get("lyrics_path", "") if has_l else "",
            "cover_path": row.get("cover_path", "") if has_c else ""
        })
        
    m_df = pd.DataFrame(matrix_rows)
    m_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Generated Modality Matrix: {OUTPUT_CSV} ({len(m_df):,} tracks)")
    
    # Print Table
    print("\n--- Per-Genre Physical Modality Availability ---")
    genre_stats = []
    for g, group in m_df.groupby("genre"):
        n = len(group)
        a_cnt = group["audio_available"].sum()
        l_cnt = group["lyrics_available"].sum()
        c_cnt = group["cover_available"].sum()
        all_cnt = (group["audio_available"] & group["lyrics_available"] & group["cover_available"]).sum()
        lc_cnt = (group["lyrics_available"] & group["cover_available"]).sum()
        al_cnt = (group["audio_available"] & group["lyrics_available"]).sum()
        ac_cnt = (group["audio_available"] & group["cover_available"]).sum()
        
        genre_stats.append({
            "Genre": g,
            "Total": n,
            "Audio": a_cnt,
            "Lyrics": l_cnt,
            "Cover": c_cnt,
            "Lyrics+Cover": lc_cnt,
            "All Three": all_cnt
        })
        print(f"  {g:<18} (N={n:>4}) | Audio: {a_cnt:>2} ({a_cnt/n*100:>4.1f}%) | Lyrics: {l_cnt:>4} ({l_cnt/n*100:>5.1f}%) | Cover: {c_cnt:>3} ({c_cnt/n*100:>5.1f}%) | L+C: {lc_cnt:>3} ({lc_cnt/n*100:>4.1f}%) | All 3: {all_cnt:>2}")

if __name__ == "__main__":
    generate_modality_matrix()

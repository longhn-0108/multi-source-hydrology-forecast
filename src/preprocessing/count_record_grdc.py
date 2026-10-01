import os
import re
import pandas as pd
from tqdm import tqdm

SEPARATOR_LINE = "#************************************************************"

###############################################################################
# Đọc sector 3 (data records)
###############################################################################
def extract_records(sector3):
    header_line_idx = None

    # Tìm dòng header chứa YYYY-MM-DD
    for i, line in enumerate(sector3):
        if re.match(r"^\s*YYYY-MM-DD", line):
            header_line_idx = i
            break

    if header_line_idx is None:
        return None

    header = sector3[header_line_idx].strip()
    cols = header.split(";")

    # Đọc các record phía sau header
    records = []
    for line in sector3[header_line_idx + 1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        parts = stripped.split(";")
        if len(parts) != len(cols):
            continue

        records.append(parts)

    # Trả về DataFrame
    df = pd.DataFrame(records, columns=cols)

    # Chuyển DATE thành datetime
    df["YYYY-MM-DD"] = pd.to_datetime(df["YYYY-MM-DD"], errors="coerce")

    return df


###############################################################################
# Tách 3 sector trong 1 file
###############################################################################
def read_file(fp):
    with open(fp, "r", encoding="utf-8") as f:
        lines = f.readlines()

    parts = []
    cur = []

    for line in lines:
        if line.strip() == SEPARATOR_LINE:
            parts.append(cur)
            cur = []
        else:
            cur.append(line)
    parts.append(cur)

    if len(parts) < 3:
        return None, None, None

    return parts[0], parts[1], parts[2]


###############################################################################
# Đếm số record từ 2020 đến nay trong 1 file
###############################################################################
def count_records_2020(fp):
    sector1, sector2, sector3 = read_file(fp)
    if sector3 is None:
        return 0

    df = extract_records(sector3)
    if df is None:
        return 0

    # Lọc từ 2020 trở đi
    df_2020 = df[df["YYYY-MM-DD"] >= pd.Timestamp("2019-01-01")]

    return len(df_2020)


###############################################################################
# Xử lý cả folder
###############################################################################
def process_folder(folder="grdc_cleaned"):
    files = [f for f in os.listdir(folder) if f.lower().endswith(".txt")]

    print(f"\n📁 Folder: {folder}")
    print(f"📌 Tìm thấy {len(files)} file GRDC.\n")

    total = 0
    per_file_stats = {}

    for fname in tqdm(files, desc="Counting records >= 2020"):
        fp = os.path.join(folder, fname)

        count = count_records_2020(fp)
        per_file_stats[fname] = count
        total += count

    print("\n===================== KẾT QUẢ =====================")
    for k, v in per_file_stats.items():
        print(f"{k}: {v} records từ 2020 → nay")

    print("----------------------------------------------------")
    print(f"🔥 Tổng toàn bộ record từ 2020 → nay: {total}")
    print("====================================================\n")

    return per_file_stats, total


###############################################################################
# RUN
###############################################################################
if __name__ == "__main__":
    process_folder("grdc_cleaned")

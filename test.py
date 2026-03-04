import json
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

POTENTIAL_PATHS = [
    "/Users/eliseomartelli/Downloads/your_instagram_activity/messages/inbox/",
    "messages/inbox/",
    "inbox/",
]

MY_NAME = "Eliseo Martelli"

EPS_SECONDS = 180
MIN_REELS_IN_BURST = 3
MIN_DURATION_MINS = 3
MAX_DURATION_MINS = 15


def load_json_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_chronobiological_prior(hour):
    if 6 <= hour < 9:
        return 0.95
    elif 9 <= hour < 13:
        return 0.50
    elif 13 <= hour < 15:
        return 0.80
    elif 15 <= hour < 22:
        return 0.30
    else:
        return 0.05


def main():
    inbox_path = None
    for path in POTENTIAL_PATHS:
        if os.path.exists(path):
            inbox_path = path
            break

    if not inbox_path:
        print("Could not find inbox path.")
        return

    json_files = glob.glob(os.path.join(inbox_path, "*", "*.json"))
    print(f"Analyzing {len(json_files)} files...")

    all_shares = []

    for jf in json_files:
        data = load_json_file(jf)
        if not data:
            continue

        participants = data.get("participants", [])
        other_participants = [
            p.get("name") for p in participants if p.get("name") != MY_NAME
        ]
        if not other_participants:
            continue

        primary_other = other_participants[0]

        for msg in data.get("messages", []):
            sender = msg.get("sender_name")
            if sender == primary_other:
                if msg.get("item_type") == "reel_share" or "share" in msg:
                    ts_ms = msg.get("timestamp_ms")
                    if ts_ms:
                        all_shares.append((primary_other, ts_ms / 1000.0))

    if not all_shares:
        print("No incoming reels found.")
        return

    df = pd.DataFrame(all_shares, columns=["sender", "timestamp"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    top_senders = df["sender"].value_counts().head(5).index.tolist()
    print(f"Top 5 senders: {top_senders}")

    anonymized_map = {
        name: f"Subject {chr(65+i)}" for i, name in enumerate(top_senders)
    }

    all_iats = []
    all_valid_events = []
    all_bursts_for_bubble = []
    global_events_for_agg = []

    for original_name in top_senders:
        subj_label = anonymized_map[original_name]
        print(f"Processing {subj_label}...")

        sender_ts = sorted(df[df["sender"] == original_name]["timestamp"].tolist())
        subj_df = df[df["sender"] == original_name].copy()
        subj_df["datetime"] = pd.to_datetime(subj_df["timestamp"], unit="s")
        subj_df["hour"] = subj_df["datetime"].dt.hour

        if len(sender_ts) > 1:
            all_iats.extend(np.diff(sender_ts))

        valid_events = []
        if len(sender_ts) >= MIN_REELS_IN_BURST:
            X = np.array(sender_ts).reshape(-1, 1)
            db = DBSCAN(eps=EPS_SECONDS, min_samples=MIN_REELS_IN_BURST).fit(X)
            labels = db.labels_

            for label in set(labels):
                if label == -1:
                    continue

                cluster_data = [
                    sender_ts[i] for i in range(len(labels)) if labels[i] == label
                ]
                start_time = pd.to_datetime(min(cluster_data), unit="s")
                end_time = pd.to_datetime(max(cluster_data), unit="s")
                duration_mins = (end_time - start_time).total_seconds() / 60.0
                num_reels = len(cluster_data)

                hour = start_time.hour
                prior = get_chronobiological_prior(hour)
                density = num_reels / duration_mins if duration_mins > 0 else num_reels
                density_score = min(1.0, density / 2.0)
                final_prob = (prior * 0.7) + (density_score * 0.3)

                if duration_mins > 0:
                    all_bursts_for_bubble.append(
                        {
                            "duration": duration_mins,
                            "reels": num_reels,
                            "prob": final_prob,
                        }
                    )

                if MIN_DURATION_MINS <= duration_mins <= MAX_DURATION_MINS:
                    event_info = {
                        "start": start_time,
                        "duration": duration_mins,
                        "reels": num_reels,
                        "probability": final_prob,
                        "day_of_week": start_time.dayofweek,
                        "hour": hour,
                    }
                    valid_events.append(event_info)
                    all_valid_events.append(event_info)

        plt.figure(figsize=(10, 4))
        plt.hist(
            subj_df["hour"],
            bins=range(25),
            align="left",
            rwidth=0.8,
            color="#4c72b0",
            alpha=0.8,
        )
        plt.xticks(range(24))
        plt.xlabel("Hour of Day")
        plt.ylabel("Frequency")
        plt.title(f"Temporal Distribution: {subj_label}")
        safe_name = subj_label.replace(" ", "_")
        plt.savefig(f"hourly_{safe_name}.png", dpi=300)
        plt.close()

        if valid_events:
            plt.figure(figsize=(12, 4))
            ev_df = pd.DataFrame(valid_events)
            plt.scatter(
                ev_df["start"],
                ev_df["probability"],
                s=ev_df["reels"] * 20,
                c=ev_df["probability"],
                cmap="viridis",
                alpha=0.7,
            )
            plt.colorbar(label="Probability Score")
            plt.title(f"Inferred Timeline: {subj_label}")
            plt.savefig(f"timeline_{safe_name}.png", dpi=300)
            plt.close()

    if all_iats:
        plt.figure(figsize=(8, 6))
        valid_iats = [t for t in all_iats if t > 0]
        bins = np.logspace(np.log10(min(valid_iats)), np.log10(max(valid_iats)), 50)
        plt.hist(valid_iats, bins=bins, density=True, color="#2c7bb6", alpha=0.7)
        plt.plot([10, 100000], [0.1, 0.00001], "r--", label="Heavy Tail")
        plt.xscale("log")
        plt.yscale("log")
        plt.title("Log-Log Distribution of IATs")
        plt.savefig("iat_loglog.png", dpi=300)
        plt.close()

    if all_valid_events:
        heatmap_data = np.zeros((7, 24))
        for ev in all_valid_events:
            heatmap_data[ev["day_of_week"], ev["hour"]] += ev["probability"]
        plt.figure(figsize=(12, 5))
        plt.imshow(heatmap_data, cmap="YlOrRd", aspect="auto")
        plt.yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        plt.title("Spatiotemporal Heatmap of Inferred Events")
        plt.savefig("heatmap_dow_hour.png", dpi=300)
        plt.close()

    if all_bursts_for_bubble:
        plt.figure(figsize=(10, 6))
        b_df = pd.DataFrame(all_bursts_for_bubble)
        plt.axvspan(
            MIN_DURATION_MINS,
            MAX_DURATION_MINS,
            color="green",
            alpha=0.1,
            label="Target Window",
        )
        plt.scatter(
            b_df["duration"],
            b_df["reels"],
            s=b_df["prob"] * 100,
            c=b_df["prob"],
            cmap="viridis",
            alpha=0.7,
        )
        plt.title("Burst Density Topography")
        plt.xlim(-1, 30)
        plt.ylim(0, b_df["reels"].max() + 2)
        plt.savefig("burst_density_bubble.png", dpi=300)
        plt.close()

    print("Running global aggregate analysis...")
    global_probs = []
    global_hours = []
    unique_senders = df["sender"].unique()

    for sender in unique_senders:
        sender_ts = sorted(df[df["sender"] == sender]["timestamp"].tolist())
        if len(sender_ts) < MIN_REELS_IN_BURST:
            continue
        X = np.array(sender_ts).reshape(-1, 1)
        db = DBSCAN(eps=EPS_SECONDS, min_samples=MIN_REELS_IN_BURST).fit(X)
        labels = db.labels_
        for label in set(labels):
            if label == -1:
                continue
            cluster_data = [
                sender_ts[i] for i in range(len(labels)) if labels[i] == label
            ]
            duration = (max(cluster_data) - min(cluster_data)) / 60.0
            if MIN_DURATION_MINS <= duration <= MAX_DURATION_MINS:
                hour = pd.to_datetime(min(cluster_data), unit="s").hour
                final_prob = (get_chronobiological_prior(hour) * 0.7) + (
                    min(1.0, (len(cluster_data) / duration) / 2.0) * 0.3
                )
                if final_prob > 0.5:
                    global_hours.append(hour)
                    global_probs.append(final_prob)

    if global_hours:
        plt.figure(figsize=(10, 5))
        plt.hist(
            global_hours,
            bins=range(25),
            align="left",
            rwidth=0.8,
            color="#55a868",
            alpha=0.8,
        )
        plt.title(f"Aggregate Chronobiological Distribution (N={len(unique_senders)})")
        plt.savefig("global_hourly_distribution.png", dpi=300)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.hist(global_probs, bins=20, color="#c44e52", alpha=0.8)
        plt.title("Distribution of Inference Confidence Scores")
        plt.savefig("inference_confidence_dist.png", dpi=300)
        plt.close()

    print("All charts generated successfully.")


if __name__ == "__main__":
    main()

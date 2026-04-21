import torch
import pandas as pd
from datetime import datetime
import os
import random
from shutil import copyfile
from glob import glob
from collections import Counter
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image




import os

# ==== LOAD TRAINED MODEL ====
model_path = os.path.join("models", "yolov8n.pt")
from ultralytics import YOLO
model = YOLO(model_path)




# ==== PATHS ====
dataset_path = os.path.join(os.path.dirname(__file__), "sample_dataset_images")
output_path = os.path.join(os.path.dirname(__file__), "output", "selected_images")
os.makedirs(output_path, exist_ok=True)




# ==== PARAMETERS ====
directions = ["north", "east", "south", "west"]
green_time_per_vehicle = 1.5  # seconds per vehicle
green_cap = 80  # max green seconds
yellow_time_fixed = 8  # yellow starts 8 seconds before green ends (overlap)
starvation_limit = 2
total_cycles = 6




# ==== PICK INITIAL IMAGES (one per direction) ====
all_images = glob(os.path.join(dataset_path, "*.jpg")) + glob(os.path.join(dataset_path, "*.png"))
if len(all_images) < 4:
    raise FileNotFoundError(
        f"Need at least 4 images in '{dataset_path}', but found {len(all_images)}. "
        "Please place your traffic images there and try again."
    )
selected_images = random.sample(all_images, 4)
direction_images = {}




for direction, img_path in zip(directions, selected_images):
    filename = os.path.basename(img_path)
    dst = os.path.join(output_path, f"{direction}_{filename}")
    copyfile(img_path, dst)
    direction_images[direction] = dst
    print(f"Assigned initial {filename} → {direction}")




print("\n✅ Initial N/S/E/W images selected successfully.\n")




# ==== UTILITY FUNCTIONS ====
def get_vehicle_details(results):
    names = model.names
    classes = [int(box.cls[0]) for box in results[0].boxes]
    type_counts = Counter(names[c] for c in classes)
    emergency_flag = any(v in ["ambulance", "fire_truck"] for v in type_counts.keys())
    return len(classes), dict(type_counts), emergency_flag




def get_avg_confidence(results):
    confidences = [float(box.conf[0]) for box in results[0].boxes]
    return sum(confidences) / len(confidences) if confidences else 0




def plot_detection(image_path, results, direction, cycle):
    img = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(10,6))
    ax.imshow(img)
    names = model.names
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        label = f"{names[cls]} {conf:.2f}"
        ax.text(x1, y1, label, color='yellow', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))
    plt.title(f"Cycle {cycle} - {direction.capitalize()} Detection")
    plt.axis('off')
    plt.show()




# ==== LOAD initial vehicle counts and flags ====
vehicle_counts = {}
vehicle_types = {}
emergency_detected = {}
avg_confidences = {}
image_indices = {d: 0 for d in directions}  # track which image index was last used




# For each direction, get all images separately (so we can pick new image later)
all_images_by_dir = {d: [] for d in directions}
random.shuffle(all_images)
for d in directions:
    all_images_by_dir[d] = random.sample(all_images, min(5,len(all_images)))  # max 5 images per dir (demo)




def update_vehicle_count_for_dir(direction, cycle):
    idx = image_indices[direction]
    images = all_images_by_dir[direction]
    img_path = images[idx % len(images)]
    image_indices[direction] = (idx + 1) % len(images)
    results = model(img_path, verbose=False)
    total, types, emergency_flag = get_vehicle_details(results)
    conf = get_avg_confidence(results)
    vehicle_counts[direction] = total
    vehicle_types[direction] = types
    emergency_detected[direction] = emergency_flag
    avg_confidences[direction] = conf
    direction_images[direction] = img_path




    # Optional: plot detection box per update (comment out if too verbose)
    # plot_detection(img_path, results, direction, cycle)
    return img_path, total




# Initialize vehicle counts, confidence, etc. for first cycle
for d in directions:
    img_path = selected_images[directions.index(d)]
    results = model(img_path, verbose=False)
    total, types, emergency_flag = get_vehicle_details(results)
    conf = get_avg_confidence(results)
    vehicle_counts[d] = total
    vehicle_types[d] = types
    emergency_detected[d] = emergency_flag
    avg_confidences[d] = conf
    direction_images[d] = img_path




# ==== STATE TRACKERS ====
skip_count = {d: 0 for d in directions}
green_history = {d: 0 for d in directions}
cumulative_red_time = {d: 0 for d in directions}
red_phase_counts = {d: 0 for d in directions}




# Track per-cycle avg confidence and wait times for plotting
confidence_per_cycle = {d: [] for d in directions}
wait_time_per_cycle = {d: [] for d in directions}




# ==== LOG data for all cycles ====
log_data = []




# ==== SIMULATION LOOP ====
for cycle_num in range(1, total_cycles + 1):
    print(f"\n🔁 Cycle {cycle_num}:")
    counts_line = " | ".join([f"{d[0].upper()}: {vehicle_counts[d]}" for d in directions])
    print("Vehicle Count:")
    print(counts_line + "\n")




    # Calculate green time per direction (1.5s per vehicle capped at green_cap)
    green_time = {}
    for d in directions:
        calc_time = math.ceil(vehicle_counts[d] * green_time_per_vehicle)
        green_time[d] = min(calc_time, green_cap) if vehicle_counts[d] > 0 else 0
        if green_time[d] < 20 and vehicle_counts[d] > 0:
            green_time[d] = 20  # minimum green time




    yellow_time = {d: yellow_time_fixed for d in directions}




    # Calculate realistic red time for each direction
    red_time = {}
    for d in directions:
        others = [o for o in directions if o != d]
        sum_green = sum(green_time[o] for o in others)
        sum_yellow = sum(yellow_time[o] for o in others)
        overlap_correction = yellow_time_fixed  # to avoid double counting
        red_time[d] = sum_green + sum_yellow - overlap_correction




    # Emergency override
    if any(emergency_detected.values()):
        green_dir = next(d for d, flag in emergency_detected.items() if flag)
        green_duration = green_time[green_dir]
        emergency_status = "Yes"
    else:
        scores = {d: vehicle_counts[d] + skip_count[d]*10 for d in directions}
        starved_dirs = [d for d, skips in skip_count.items() if skips >= starvation_limit]
        if starved_dirs:
            green_dir = max(starved_dirs, key=lambda d: scores[d])
            print(f"⚖️ Fairness: Forcing green for {green_dir.upper()} due to starvation.")
            skip_count[green_dir] = 0
        else:
            green_dir = max(scores, key=scores.get)




        green_duration = green_time[green_dir]
        emergency_status = "No"




    # Find second highest excluding green_dir
    sorted_dirs = sorted(vehicle_counts.items(), key=lambda x: x[1], reverse=True)
    second_high_dir = None
    for d, _ in sorted_dirs:
        if d != green_dir:
            second_high_dir = d
            break




    # Prepare signal timing strings
    signal_colors = {}
    signal_colors[green_dir] = f"🟢 {green_time[green_dir]}s"
    signal_colors[second_high_dir] = f"🟡 {yellow_time[second_high_dir]}s"
    other_dirs = [d for d in directions if d not in [green_dir, second_high_dir]]
    for d in other_dirs:
        signal_colors[d] = f"🔴 {red_time[d]}s"




    # Update skip counts and red times
    for d in directions:
        if d == green_dir:
            skip_count[d] = 0
            green_history[d] += 1
        else:
            skip_count[d] += 1
            cumulative_red_time[d] += green_time[d] + yellow_time[d]
            red_phase_counts[d] += 1




    # Save confidence & wait time for plotting
    for d in directions:
        confidence_per_cycle[d].append(avg_confidences[d])
        if red_phase_counts[d] > 0:
            avg_wait = cumulative_red_time[d] / red_phase_counts[d]
        else:
            avg_wait = 0
        wait_time_per_cycle[d].append(avg_wait)




    # Print signal timing & decision
    print("Signal Timing:")
    print(" | ".join([f"{d.capitalize()}: {signal_colors[d]}" for d in directions]) + "\n")
    print("Decision:")
    print(f"{green_dir.upper()} had the highest traffic → assigned green")
    print(f"{second_high_dir.upper()} second highest → yellow")
    print(f"{', '.join([d.upper() for d in other_dirs])} will get priority in the next cycle due to fairness/starvation.\n")




    # Update vehicle count/image for green_dir if green time < cap
    if green_time[green_dir] < green_cap:
        new_img, new_count = update_vehicle_count_for_dir(green_dir, cycle_num)
        print(f"Updated {green_dir.upper()} image for next cycle: {os.path.basename(new_img)} with vehicle count: {new_count}")
    else:
        print(f"{green_dir.upper()} green time capped at {green_cap}s, vehicle count assumed unchanged for next cycle.")




    # Log cycle data
    log_data.append({
        "Cycle": cycle_num,
        **{f"Count_{d}": vehicle_counts[d] for d in directions},
        "Green_Direction": green_dir,
        "Green_Duration": green_time[green_dir],
        "Yellow_Duration": yellow_time[second_high_dir],
        **{f"Red_Duration_{d}": red_time[d] for d in directions},
        "Emergency": emergency_status,
        "Skip_Counts": skip_count.copy(),
        **{f"Confidence_{d}": avg_confidences[d] for d in directions},
    })




# ===== Save log CSV =====
csv_path = os.path.join(output_path, "traffic_simulation_log.csv")
df = pd.DataFrame(log_data)
df.to_csv(csv_path, index=False)
print(f"\n📊 Simulation log saved to: {csv_path}")




# ===== PLOT YOLO CONFIDENCE TREND =====
plt.figure(figsize=(10, 6))
for d in directions:
    plt.plot(range(1, total_cycles + 1), confidence_per_cycle[d], marker='o', label=d.capitalize())
plt.title('YOLO Detection Average Confidence per Cycle')
plt.xlabel('Cycle Number')
plt.ylabel('Average Confidence')
plt.legend()
plt.grid(True)
plt.show()

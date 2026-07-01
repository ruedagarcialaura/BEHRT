import matplotlib.pyplot as plt

# 1. Read the log file
with open('visualizations/clean_logs.txt', 'r') as f:
    lines = f.readlines()

# Lists to store the extracted values
cnts = []
losses = []
precisions = []

# Variables to create a continuous global counter
offset = 0
prev_cnt = 0

# 2. Extract the values from the log lines
for line in lines:
    if "Loss:" in line and "precision:" in line:
        try:
            parts = line.split('|')
            cnt = int(parts[1].split(':')[1].strip())
            loss = float(parts[2].split(':')[1].strip())
            prec = float(parts[3].split(':')[1].strip())
            
            # --- THE MAGIC OF THE X-AXIS ---
            # If the current cnt is less than the previous one, it means we changed Epochs
            if cnt < prev_cnt:
                offset += prev_cnt # Save the accumulated steps so far
                
            # The actual global step is the current cnt plus everything accumulated in previous epochs
            global_step = cnt + offset
            cnts.append(global_step)
            prev_cnt = cnt
            
            losses.append(loss)
            precisions.append(prec)
        except:
            continue # Ignore warnings

# --- THE MAGIC OF THE Y-AXIS (Smoothing) ---
def smooth_curve(data, weight=0.85):
    """Applies an exponential smoothing so the graph doesn't have such aggressive spikes"""
    smoothed = []
    last = data[0]
    for point in data:
        smooth_val = last * weight + (1 - weight) * point
        smoothed.append(smooth_val)
        last = smooth_val
    return smoothed

smoothed_losses = smooth_curve(losses, weight=0.90)
smoothed_precisions = smooth_curve(precisions, weight=0.90)

# 3. Draw the Loss graph
plt.figure(figsize=(10, 5))
# Draw the original line lighter in the background (optional)
plt.plot(cnts, losses, color='red', alpha=0.2, label='Raw Data')
# Draw the smoothed and continuous line
plt.plot(cnts, smoothed_losses, color='red', linewidth=2, label='Training Loss (Smoothed)')
plt.title('Learning Curve: Loss', fontsize=14, fontweight='bold')
plt.xlabel('Global Training Steps', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# 4. Draw the Precision graph
plt.figure(figsize=(10, 5))
plt.plot(cnts, precisions, color='blue', alpha=0.2, label='Raw Data')
plt.plot(cnts, smoothed_precisions, color='blue', linewidth=2, label='Masked Code Precision (Smoothed)')
plt.title('Learning Curve: Precision', fontsize=14, fontweight='bold')
plt.xlabel('Global Training Steps', fontsize=12)
plt.ylabel('Precision (0 to 1)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
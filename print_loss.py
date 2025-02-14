import matplotlib.pyplot as plt

# Load data from text files
files = [
    "/home/coolbot/data/loss_1view_cam_3.txt",
    "/home/coolbot/data/loss_2view_cam_3.txt",
    "/home/coolbot/data/loss_3view_cam_3.txt"
]

loss_data = {"j2d": [], "surface": []}

# Read losses from each file
for file in files:
    j2d_losses = []
    surface_losses = []
    with open(file, "r") as f:
        for line in f:
            if "j2d loss" in line:
                j2d_losses.append(float(line.split()[-1]))
            elif "surface loss" in line:
                surface_losses.append(float(line.split()[-1]))
    loss_data["j2d"].append(j2d_losses)
    loss_data["surface"].append(surface_losses)

frames = range(len(loss_data["j2d"][0]))

# Plot j2d loss
plt.figure(figsize=(10, 5))
for i, losses in enumerate(loss_data["j2d"]):
    plt.plot(frames, losses, label=f'{i+1} View ')
plt.xlabel("Frame")
plt.ylabel("j2d Loss")
plt.title("j2d Loss Across Different Views")
plt.legend()
plt.grid(True)
plt.show()

# Plot surface loss
plt.figure(figsize=(10, 5))
for i, losses in enumerate(loss_data["surface"]):
    plt.plot(frames, losses, label=f'{i+1} View ')
plt.xlabel("Frame")
plt.ylabel("Surface Loss")
plt.title("Surface Loss Across Different Views")
plt.legend()
plt.grid(True)
plt.show()

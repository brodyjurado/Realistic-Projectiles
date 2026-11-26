from numpy import *
import matplotlib.pyplot as plt

step = 0.01
g = 1.0

shapes = {
    "Standard (k=1)": (1.0, 0.0),
    "Football": (0.45, 0.18),
    "Golf Ball": (0.60, 0.35),
    "Paper airplane": (0.75, 1.15),
    "Shuttlecock": (2.40, 0.0)
}

def f(t, u, kd, kl):
    x = u[0]
    y = u[1]
    vx = u[2]
    vy = u[3]
    v = sqrt(vx**2 + vy**2)
    ax = -kd * v * vx - kl * v * vy
    ay = -g - kd * v * vy + kl * v * vx
    return [vx, vy, ax, ay]

def RK4MethodStep(t, u, step, kd, kl):
    k1 = f(t, u, kd, kl)
    k2 = f(t + step/2, add(u, multiply(step/2, k1)), kd, kl)
    k3 = f(t + step/2, add(u, multiply(step/2, k2)), kd, kl)
    k4 = f(t + step, add(u, multiply(step, k3)), kd, kl)
    return add(u, multiply(step/6, add(add(add(k1, multiply(2, k2)), multiply(2, k3)), k4)))

def simulate_projectile(v0, angle_deg, kd=1.0, kl=0.0, step=0.01):
    theta = radians(angle_deg)
    t_initial = 0.0
    u_initial = [0.0, 0.0, v0 * cos(theta), v0 * sin(theta)]
    t_values = [t_initial]
    u_values = [u_initial]
    
    while len(u_values) < 2 or u_values[-1][1] >= 0:
        u_next = RK4MethodStep(t_values[-1], u_values[-1], step, kd, kl)
        u_values.append(u_next)
        t_values.append(t_values[-1] + step)
        if t_values[-1] > 200:
            break
            
    data = transpose(u_values)
    x_values = data[0]
    y_values = data[1]
    vx_values = data[2]
    vy_values = data[3]
    
    if y_values[-1] < 0 and len(y_values) >= 2:
        x1, y1 = x_values[-2], y_values[-2]
        x2, y2 = x_values[-1], y_values[-1]
        firing_range = x1 - y1 * (x2 - x1) / (y2 - y1)
    else:
        firing_range = x_values[-1]
        
    return array(t_values), x_values, y_values, vx_values, vy_values, firing_range

#Coupled motion
plt.figure(1, figsize=(8, 5))
vx0_fixed = 1.0
for vy0_test in [0.5, 1.0, 2.5, 4.0]:
    v0_test = sqrt(vx0_fixed**2 + vy0_test**2)
    ang_test = degrees(arctan2(vy0_test, vx0_fixed))
    t_vals, x_vals, y_vals, vx_vals, vy_vals, _ = simulate_projectile(v0_test, ang_test, 1.0, 0.0, step)
    plt.plot(t_vals, vx_vals, label=f"vy0 = {vy0_test}")
plt.title("Horizontal Velocity Decay for Different Initial Vertical Speeds (vx0 = 1.0)")
plt.xlabel("Time (t)")
plt.ylabel("Horizontal Velocity (vx)")
plt.legend()
plt.grid(True)

#Trajectories
plt.figure(2, figsize=(8, 5))
for v0 in [0.5, 1.5, 3.0]:
    for ang in (20, 45, 70):
        _, x_vals, y_vals, _, _, _ = simulate_projectile(v0, ang, 1.0, 0.0, step)
        plt.plot(x_vals, y_vals, label=f"v0={v0}, angle={ang}°")
plt.ylim(bottom=0)
plt.title("Projectile Trajectories Across Initial Speeds and Firing Angles")
plt.xlabel("Horizontal Distance (x)")
plt.ylabel("Vertical Height (y)")
plt.legend(fontsize=8, ncol=2)
plt.grid(True)

#Range vs Angle
plt.figure(3, figsize=(8, 5))
angles = linspace(5, 85, 41)
for v0 in [0.5, 1.0, 2.0, 4.0]:
    ranges = []
    for ang in angles:
        _, _, _, _, _, r = simulate_projectile(v0, ang, 1.0, 0.0, step)
        ranges.append(r)
    best_idx = argmax(ranges)
    plt.plot(angles, ranges, label=f"v0 = {v0} (max at {angles[best_idx]:.1f}°)")
plt.title("Firing Range vs. Firing Angle (k = 1, No Lift)")
plt.xlabel("Firing Angle (degrees)")
plt.ylabel("Firing Range (x)")
plt.legend()
plt.grid(True)

#Range vs Speed
plt.figure(4, figsize=(8, 5))
speeds = linspace(0.2, 5.0, 35)
for ang in (15, 30, 45, 60):
    ranges = []
    for v0 in speeds:
        _, _, _, _, _, r = simulate_projectile(v0, ang, 1.0, 0.0, step)
        ranges.append(r)
    plt.plot(speeds, ranges, label=f"Angle = {ang}°")
plt.title("Firing Range vs. Initial Speed (k = 1, No Lift)")
plt.xlabel("Initial Speed (v0)")
plt.ylabel("Firing Range (x)")
plt.legend()
plt.grid(True)

#Projectile shapes
plt.figure(5, figsize=(8, 5))
for name, (kd_val, kl_val) in shapes.items():
    _, x_vals, y_vals, _, _, r = simulate_projectile(2.0, 35.0, kd_val, kl_val, step)
    plt.plot(x_vals, y_vals, label=f"{name} (Range={r:.2f})")
plt.ylim(bottom=0)
plt.title("Trajectories by Projectile Shape (v0 = 2.0, Angle = 35°)")
plt.xlabel("Horizontal Distance (x)")
plt.ylabel("Vertical Height (y)")
plt.legend()
plt.grid(True)

#Lift optimization
plt.figure(6, figsize=(8, 5))
kl_values = linspace(0.0, 2.5, 51)
test_cases = [(1.0, 30), (1.0, 45), (2.5, 20), (2.5, 45)]
print("Optimal Lift Constants (k_L) for Maximum Range:")
for v0, ang in test_cases:
    ranges = []
    for kl_test in kl_values:
        _, _, _, _, _, r = simulate_projectile(v0, ang, 1.0, kl_test, step)
        ranges.append(r)
    best_idx = argmax(ranges)
    best_kl = kl_values[best_idx]
    max_r = ranges[best_idx]
    print(f"v0 = {v0}, angle = {ang}° -> Optimal k_L = {best_kl:.2f}, Max Range = {max_r:.3f}")
    plt.plot(kl_values, ranges, label=f"v0={v0}, angle={ang}° (opt kL={best_kl:.2f})")
plt.title("Firing Range vs. Lift Constant (k_L) with Quadratic Drag (k_d = 1)")
plt.xlabel("Lift Constant (k_L)")
plt.ylabel("Firing Range (x)")
plt.legend()
plt.grid(True)

plt.show()
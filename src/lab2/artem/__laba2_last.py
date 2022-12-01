x_1, y_1 = self.object_1.math_exp
x_2, y_2 = self.object_2.math_exp
mid_x = (x_1 + x_2)/2
mid_y = (y_1 + y_2)/2
self.ax.scatter([ [mid_x] ], [ mid_y ], color='#FDB94D')


x = lambda grad: cos(grad_to_rad(grad))  + mid_x
y = lambda grad: sin(grad_to_rad(grad))  + mid_y


ang = degrees(acos( (mid_x - x_1) / sqrt((mid_x - x_1)**2 + (mid_y - y_1)**2) ))

normal_ang = 90 + ang
normal_x = x(normal_ang)
normal_y = y(normal_ang)

# y = kx + b
k = lambda p_1, p_2: (p_2[1] - p_1[1]) / (p_2[0] - p_1[0])  # p_N = x_n, y_n; # p_2 = x_2, y_2;
b = lambda p_1, p_2: (p_1[1] * p_2[0] - p_1[0] * p_2[1] ) / (p_2[0] - p_1[0])

k_normal = k( (mid_x, mid_y), (normal_x, normal_y) )
b_normal = b( (mid_x, mid_y), (normal_x, normal_y) )

# print(f"normal_ang = { normal_ang } или { 180 + degrees(atan(k_normal)) }")
# print(f"y = {round(k_normal, 3)} * x + {b_normal}")

x_points = list(map(lambda value: value[0], list(self.object_1.points) + list(self.object_2.points)))
x_min, x_max = round(np.min(x_points), 2), round(np.max(x_points), 2)

# print(f"[{x_min},{x_max}]\n")

new_y_norm_left_point = k_normal * x_min + b_normal
new_y_norm_right_point = k_normal * x_max + b_normal
lM = mlines.Line2D(
    [x_min, x_max], [new_y_norm_left_point, new_y_norm_right_point], 
    color="#000", linestyle="--", marker="x"
)
self.ax.add_line(lM)
# print(f"[{x_min, x_max}], [{new_y_norm_left_point, new_y_norm_right_point}]")

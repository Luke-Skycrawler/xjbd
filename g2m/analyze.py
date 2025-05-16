import warp as wp
import numpy as np
@wp.func 
def value_of_quadratic_surface_2d(x: float, y: float, A: float, B: float, C: float, D: float, E: float, F: float) -> float:
    return A * x * x + B * x * y + C * y * y + D * x + E * y + F

@wp.func
def compute_distance_cone_cone(c0: wp.vec3, c1: wp.vec3, c2: wp.vec3, c3: wp.vec3, 
                     r0: float, r1: float, r2: float, r3: float):
    
    sC1 = c0 - c1
    sC2 = c3 - c2
    sC3 = c1 - c3

    sR1 = r0 - r1
    sR2 = r2 - r3
    sR3 = r1 + r3

    A = wp.dot(sC1, sC1) - sR1 * sR1
    B = 2.0 * (wp.dot(sC1, sC2) - sR1 * sR2)
    C = wp.dot(sC2, sC2) - sR2 * sR2
    D = 2.0 * (wp.dot(sC1, sC3) - sR1 * sR3)
    E = 2.0 * (wp.dot(sC2, sC3) - sR2 * sR3)
    F = wp.dot(sC3, sC3) - sR3 * sR3

    delta = 4.0 * A * C - B * B

    alpha = float(0.0)
    beta = float(0.0)
    distance = float(0.0)
    mode = int(0)
    distance = value_of_quadratic_surface_2d(alpha, beta, A, B, C, D, E, F)

    # Evaluate each test case manually (no lists)
    temp_alpha = 1.0
    temp_beta = 0.0
    temp_dist = value_of_quadratic_surface_2d(temp_alpha, temp_beta, A, B, C, D, E, F)
    if temp_dist < distance:
        distance = temp_dist
        alpha = temp_alpha
        beta = temp_beta

    temp_alpha = 0.0
    temp_beta = 1.0
    temp_dist = value_of_quadratic_surface_2d(temp_alpha, temp_beta, A, B, C, D, E, F)
    if temp_dist < distance:
        distance = temp_dist
        alpha = temp_alpha
        beta = temp_beta

    temp_alpha = 1.0
    temp_beta = 1.0
    temp_dist = value_of_quadratic_surface_2d(temp_alpha, temp_beta, A, B, C, D, E, F)
    if temp_dist < distance:
        distance = temp_dist
        alpha = temp_alpha
        beta = temp_beta

    if C != 0.0:
        temp_alpha = 0.0
        temp_beta = -E / (2.0 * C)
        if 0.0 <= temp_beta <= 1.0:
            temp_dist = value_of_quadratic_surface_2d(temp_alpha, temp_beta, A, B, C, D, E, F)
            if temp_dist < distance:
                distance = temp_dist
                alpha = temp_alpha
                beta = temp_beta
                mode = 1
                

        temp_alpha = 1.0
        temp_beta = -(B + E) / (2.0 * C)
        if 0.0 <= temp_beta <= 1.0:
            temp_dist = value_of_quadratic_surface_2d(temp_alpha, temp_beta, A, B, C, D, E, F)
            if temp_dist < distance:
                distance = temp_dist
                alpha = temp_alpha
                beta = temp_beta
                mode = 2

    if A != 0.0:
        temp_alpha = -D / (2.0 * A)
        temp_beta = 0.0
        if 0.0 <= temp_alpha <= 1.0:
            temp_dist = value_of_quadratic_surface_2d(temp_alpha, temp_beta, A, B, C, D, E, F)
            if temp_dist < distance:
                distance = temp_dist
                alpha = temp_alpha
                beta = temp_beta
                mode = 3

        temp_alpha = -(B + D) / (2.0 * A)
        temp_beta = 1.0
        if 0.0 <= temp_alpha <= 1.0:
            temp_dist = value_of_quadratic_surface_2d(temp_alpha, temp_beta, A, B, C, D, E, F)
            if temp_dist < distance:
                distance = temp_dist
                alpha = temp_alpha
                beta = temp_beta
                mode = 4

    if delta != 0.0:
        temp_alpha = (B * E - 2.0 * C * D) / delta
        temp_beta = (B * D - 2.0 * A * E) / delta
        if 0.0 <= temp_alpha <= 1.0 and 0.0 <= temp_beta <= 1.0:
            temp_dist = value_of_quadratic_surface_2d(temp_alpha, temp_beta, A, B, C, D, E, F)
            if temp_dist < distance:
                distance = temp_dist
                alpha = temp_alpha
                beta = temp_beta
                mode = 5

    # Compute closest points
    cp = alpha * c0 + (1.0 - alpha) * c1
    cq = beta * c2 + (1.0 - beta) * c3
    rp = alpha * r0 + (1.0 - alpha) * r1
    rq = beta * r2 + (1.0 - beta) * r3

    dir = cq - cp
    dir = wp.normalize(dir)

    cloest_p0 = cp + dir * rp
    cloest_p1 = cq - dir * rq
    return distance, cloest_p0, cloest_p1, mode


@wp.func
def compute_distance_slab_sphere(c0: wp.vec3, c1: wp.vec3, c2: wp.vec3, c3: wp.vec3, 
                     r0: float, r1: float, r2: float, r3: float):
    '''
    c0, c1, c2 is the slab and c3 is the sphere
    '''
    sC1 = c0 - c2
    sC2 = c1 - c2
    sC3 = c2 - c3

    sR1 = r0 - r2
    sR2 = r1 - r2
    sR3 = r2 + r3

    A = wp.dot(sC1, sC1) - sR1 * sR1
    B = 2.0 * (wp.dot(sC1, sC2) - sR1 * sR2)
    C = wp.dot(sC2, sC2) - sR2 * sR2
    D = 2.0 * (wp.dot(sC1, sC3) - sR1 * sR3)
    E = 2.0 * (wp.dot(sC2, sC3) - sR2 * sR3)
    F = wp.dot(sC3, sC3) - sR3 * sR3

    delta = 4.0 * A * C - B * B

    alpha = float(0.0)
    beta = float(0.0)
    distance = float(0.0)
    mode =int(0)
    distance = value_of_quadratic_surface_2d(alpha, beta, A, B, C, D, E, F)

    # Check multiple cases for alpha and beta manually
    temp_alpha = 1.0
    temp_beta = 0.0
    temp_dist = value_of_quadratic_surface_2d(temp_alpha, temp_beta, A, B, C, D, E, F)
    if temp_dist < distance:
        distance = temp_dist
        alpha = temp_alpha
        beta = temp_beta

    temp_alpha = 0.0
    temp_beta = 1.0
    temp_dist = value_of_quadratic_surface_2d(temp_alpha, temp_beta, A, B, C, D, E, F)
    if temp_dist < distance:
        distance = temp_dist
        alpha = temp_alpha
        beta = temp_beta

    if C != 0.0:
        temp_alpha = 0.0
        temp_beta = -E / (2.0 * C)
        if 0.0 <= temp_beta <= 1.0:
            temp_dist = value_of_quadratic_surface_2d(temp_alpha, temp_beta, A, B, C, D, E, F)
            if temp_dist < distance:
                distance = temp_dist
                alpha = temp_alpha
                beta = temp_beta
                mode = 1
    if A != 0.0:
        temp_alpha = -D / (2.0 * A)
        temp_beta = 0.0
        if 0.0 <= temp_alpha <= 1.0:
            temp_dist = value_of_quadratic_surface_2d(temp_alpha, temp_beta, A, B, C, D, E, F)
            if temp_dist < distance:
                distance = temp_dist
                alpha = temp_alpha
                beta = temp_beta
                mode = 2

    if (A - B + C) != 0.0:
        temp_alpha = 0.5 * (2.0 * C + E - B - D) / (A - B + C)
        temp_beta = 1.0 - temp_alpha
        if 0.0 <= temp_alpha <= 1.0:
            temp_dist = value_of_quadratic_surface_2d(temp_alpha, temp_beta, A, B, C, D, E, F)
            if temp_dist < distance:
                distance = temp_dist
                alpha = temp_alpha
                beta = temp_beta
                mode = 3

    if delta != 0.0:
        temp_alpha = (B * E - 2.0 * C * D) / delta
        temp_beta = (B * D - 2.0 * A * E) / delta
        if 0.0 <= temp_alpha <= 1.0 and 0.0 <= temp_beta <= 1.0 and temp_alpha + temp_beta < 1.0:
            temp_dist = value_of_quadratic_surface_2d(temp_alpha, temp_beta, A, B, C, D, E, F)
            if temp_dist < distance:
                distance = temp_dist
                alpha = temp_alpha
                beta = temp_beta
                mode = 3

    # Compute closest points
    cp = alpha * c0
    cq = c3  # Spheres[3].center remains fixed

    closest_p0 = cp
    closest_p1 = cq
    return distance, closest_p0, closest_p1, mode

@wp.kernel
def test_kernel(c: wp.array(dtype = wp.vec3), radius: wp.array(dtype = float), dist: wp.array(dtype = float), modes: wp.array(dtype = int)):
    i = wp.tid()
    c0 = c[i * 4 + 0]
    c1 = c[i * 4 + 1]
    c2 = c[i * 4 + 2]
    c3 = c[i * 4 + 3]

    r0 = radius[i * 4 + 0]
    r1 = radius[i * 4 + 1]
    r2 = radius[i * 4 + 2]
    r3 = radius[i * 4 + 3]

    d, p0, p1, m = compute_distance_slab_sphere(c0, c1, c2, c3, r0, r1, r2, r3)
    dist[i] = d

    modes[i] = m


if __name__ == "__main__":
    wp.config.max_unroll = 0
    wp.init()
    n_samples = 1000
    samples = np.random.rand(n_samples * 4, 3)
    radius =  np.random.rand(n_samples * 4) * 1e-2

    c = wp.zeros(n_samples * 4, dtype =wp.vec3)
    r = wp.zeros(n_samples * 4, dtype = float)  
    m = wp.zeros(n_samples, dtype = int)
    c.assign(samples)
    r.assign(radius)
    dist = wp.zeros(n_samples, dtype = float)
    # print(samples, radius)
    wp.launch(test_kernel, n_samples, inputs = [c, r, dist, m])
    
    from mtk import ConeConeConstraint, SlabSphereConstraint
    v4 = np.zeros((n_samples * 4, 4), dtype = float)
    v4[:, :3] = samples
    v4[:, 3] = radius
    d_ref = np.zeros(n_samples, dtype = float)
    modes_ref = np.zeros(n_samples, dtype = int)
    for i in range(n_samples):
        # constraint = ConeConeConstraint(
        constraint = SlabSphereConstraint(
            v4[i * 4 + 0], 
            v4[i * 4 + 1], 
            v4[i * 4 + 2], 
            v4[i * 4 + 3])
        d_ref[i] = constraint.compute_distance()
        mode = constraint.get_distance_mode()
        modes_ref[i] =  mode

    if n_samples < 10:
        print(dist.numpy(), m.numpy())
        print(f"ref = {d_ref}, modes = {modes_ref}")
    else:
        diff = d_ref - dist.numpy()
        diff_mode = modes_ref - m.numpy()
        errors = np.arange(n_samples)[diff_mode != 0]
        
        print(f"diff = {np.max(np.abs(diff))}, diff_mode = {errors}")
    
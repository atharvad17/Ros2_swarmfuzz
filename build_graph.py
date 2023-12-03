import numpy as np

def build_graph(dev, goal, pos_att):
    # Initialization
    nb = len(pos_att[0])
    v = np.ones(nb)
    A = np.ones((nb, nb)) - np.diag(v)  # Initialize the adjacent matrix
    A_zero = np.zeros((nb, nb))

    for i in range(nb):
        for j in range(nb):
            if i == j:
                continue  # Victim drone and attack drone should be different

            if dev == goal[j]:  # Deviation has positive influence
                # E.g., the attack drone deviates to its right side (dev = 1).
                # This helps other drones also deviate to its right side,
                # regardless of the relative location between the attack drone and other drones.
                # Thereby, if the target direction of the victim drone is also right (goal(j) = 1),
                # the deviation of the attack drone has positive influence,
                # and we add an edge with the calculated weight in the graph.
                delta_y = pos_att[1, i] - pos_att[1, j]
                delta_x = pos_att[0, i] - pos_att[0, j]
                A[j, i] = abs(delta_y / delta_x) if delta_x != 0 else 0  # Calculate the weight
            else:  # Deviation has no positive influence
                A[j, i] = 0

    flag = 1 if not np.array_equal(A, A_zero) else 0

    return A, flag

# Example usage:
# dev = 1  # right deviation
# goal = [1, -1, 1, -1]  # target directions of drones under attack
# pos_att = np.array([[1, 2, 3, 4], [1, 2, 1, 2]])  # drones' location at initial attack start time
# A, flag = build_graph(dev, goal, pos_att)
# print("Generated Swarm Vulnerability Graph:")
# print(A)
# print("Flag: {}".format(flag))

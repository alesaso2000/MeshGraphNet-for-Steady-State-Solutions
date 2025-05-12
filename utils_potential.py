import numpy as np


def body_panels(xy):
    xpan = np.concatenate([xy[0:-1, 0:1], xy[1:, 0:1]], axis=1)
    ypan = np.concatenate([xy[0:-1, 1:2], xy[1:, 1:2]], axis=1)
    dx = np.diff(xpan, axis=1)
    dy = np.diff(ypan, axis=1)
    thpan = np.arctan2(dy, dx)
    xycpan = 0.5*np.concatenate([np.sum(xpan, axis=1, keepdims=True),
                                  np.sum(ypan, axis=1, keepdims=True)], axis=1)
    nor = np.concatenate([-np.sin(thpan), np.cos(thpan)], axis=1)
    return xycpan, xpan, ypan, thpan, nor



def mk_coeff_local_body(xycpan, thpan, xpan, ypan):
    npan = len(thpan)
    Anor = np.zeros((npan, npan))
    Atan = np.zeros((npan, npan))
    for ip in range(npan):
        ising = np.zeros((npan, 1))
        ising[ip] = 1
        xyp = np.concatenate([xpan[ip, 0:2].reshape(-1, 1), ypan[ip, 0:2].reshape(-1, 1)], axis=1)
        thp = thpan[ip]
        AA = source_coeff_vel_local(xycpan, thpan, xyp, thp, ising)
        Anor[:, ip:ip+1] = AA[:, 1:2]
        Atan[:, ip:ip+1] = AA[:, 0:1]
    return Anor, Atan


# def mk_coeff_xy_body(xy, thpan, xpan, ypan):
#     npan = len(thpan)
#     Ax = np.zeros((npan, npan))
#     Ay = np.zeros((npan, npan))
#     for ip in range(npan-1):
#         ising = np.zeros((npan, 1))
#         ising[ip] = 1
#         xyp = np.concatenate([xpan[ip, 0:2].reshape(-1, 1), ypan[ip, 0:2].reshape(-1, 1)], axis=1)
#         thp = thpan[ip]
#         AA = source_coeff_vel_bodyfixed(xy, xyp, thp, ising)
#         Ay[:, ip:ip + 1] = AA[:, 1:2]
#         Ax[:, ip:ip + 1] = AA[:, 0:1]
#     return Ax, Ay


def mk_coeff_xy(xy, thpan, xpan, ypan):
    npan = len(thpan)
    Ax = np.zeros((len(xy), npan))
    Ay = np.zeros((len(xy), npan))
    ising = np.zeros((len(xy), 1))
    for ip in range(npan-1):
        xyp = np.concatenate([xpan[ip, 0:2].reshape(-1, 1), ypan[ip, 0:2].reshape(-1, 1)], axis=1)
        thp = thpan[ip]
        AA = source_coeff_vel_bodyfixed(xy, xyp, thp, ising)
        Ay[:, ip:ip + 1] = AA[:, 1:2]
        Ax[:, ip:ip + 1] = AA[:, 0:1]
    return Ax, Ay




def source_coeff_vel_local(xycpan, thpan, xyp, thp, ising):

    lc1 = xycpan - xyp[0, 0:2]
    lc2 = xycpan - xyp[1, 0:2]

    rij = np.sqrt(np.sum(lc1**2, axis=1, keepdims=True))
    rijp1 = np.sqrt(np.sum(lc2**2, axis=1, keepdims=True))

    colp_sin = np.sin(thpan)
    colp_cos = np.cos(thpan)
    p_sin = np.sin(thp)
    p_cos = np.cos(thp)

    betaij = np.arctan2(lc2[:, 1:2]*lc1[:, 0:1] - lc2[:, 0:1]*lc1[:, 1:2],
                           lc2[:, 0:1]*lc1[:, 0:1] + lc2[:, 1:2]*lc1[:, 1:2])

    lr2r1 = np.log(rijp1/rij)

    inv2pi = 1/2/np.pi

    A0 = inv2pi * (
        (
            betaij * (colp_sin*p_cos -
                      colp_cos*p_sin) -
            lr2r1 * (colp_cos*p_cos +
                     colp_sin*p_sin)
        ) * (1 - ising) +
        (
            np.pi * (colp_sin*p_cos -
                        colp_cos*p_sin)
        ) * ising
    )

    A1 = inv2pi * (
            (
                    betaij * (colp_cos * p_cos +
                              colp_sin * p_sin) +
                    lr2r1 * (colp_sin * p_cos -
                             colp_cos * p_sin)
            ) * (1 - ising) +
            (
                    np.pi * (colp_cos * p_cos +
                                colp_sin * p_sin)
            ) * ising
    )

    AA = np.concatenate([A0, A1], axis=1)
    return AA


def source_coeff_vel_bodyfixed(xycpan, xyp, thp, ising):

    lc1 = xycpan - xyp[0, 0:2]
    lc2 = xycpan - xyp[1, 0:2]

    rij = np.sqrt(np.sum(lc1**2, axis=1, keepdims=True))
    rijp1 = np.sqrt(np.sum(lc2**2, axis=1, keepdims=True))

    p_sin = np.sin(thp)
    p_cos = np.cos(thp)

    betaij = np.arctan2(lc2[:, 1:2]*lc1[:, 0:1] - lc2[:, 0:1]*lc1[:, 1:2],
                           lc2[:, 0:1]*lc1[:, 0:1] + lc2[:, 1:2]*lc1[:, 1:2])

    '''
    lr2r1 = torch.tensor([-1, 1]).to(params['device'])
    for i in range(len(rij)):
        if (rij[i] > 0) & (rijp1[i] > 0):
            lr2r1 = torch.cat([lr2r1, torch.log(rijp1[i] / rij[i])], dim=0)
        else:
            lr2r1 = torch.cat([lr2r1, torch.tensor([0]).to(params['device'])], dim=0)
    '''

    lr2r1 = np.log(rijp1/rij)

    inv2pi = 1/2/np.pi

    A0 = inv2pi * (
        (
            betaij * (- p_sin) -
            lr2r1 * (+ p_cos)
        ) * (1 - ising) +
        (
            np.pi * (- p_sin)
        ) * ising
    )

    A1 = inv2pi * (
            (
                    betaij * (+ p_cos) +
                    lr2r1 * (- p_sin)
            ) * (1 - ising) +
            (
                    np.pi * (+ p_cos)
            ) * ising
    )

    AA = np.concatenate([A0, A1], axis=1)
    return AA


def sol_pot(xy, xpan, ypan, thpan, norpan):
    An, _ = mk_coeff_local_body(xy, thpan, xpan, ypan)
    B = norpan[:, 0:1]
    sigmax = np.linalg.solve(An, B)
    B = norpan[:, 1:2]
    sigmay = np.linalg.solve(An, B)
    return sigmax, sigmay




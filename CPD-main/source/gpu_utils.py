import cupy as cp
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay
from numba import njit

A_EPS = 1e-12
K_BARRIER = 1e-6


def _as_tri_param_gpu(param, n_tri):
    arr = cp.asarray(param, dtype=cp.float64)
    if arr.ndim == 0:
        return cp.full((n_tri,), arr.item(), dtype=cp.float64)
    arr = arr.reshape(-1)
    if arr.size == 1:
        return cp.full((n_tri,), float(arr[0].item()), dtype=cp.float64)
    if arr.size != n_tri:
        raise ValueError(f"Expected parameter size {n_tri}, got {arr.size}")
    return arr.astype(cp.float64, copy=False)


def delaunay_triangulate(pos:NDArray)->NDArray:
    tri = Delaunay(pos).simplices
    return tri

def calc_ele_areas_gpu(pos:NDArray, tri:NDArray)->NDArray:
    p1 = pos[tri[:,0]]
    p2 = pos[tri[:,1]]
    p3 = pos[tri[:,2]]
    areas = 0.5*cp.abs(p1[:,0]*(p2[:,1]-p3[:,1])+p2[:,0]*(p3[:,1]-p1[:,1])+p3[:,0]*(p1[:,1]-p2[:,1]))
    return areas

def calc_particle_areas_gpu(pos:NDArray, tri:NDArray, areas:NDArray)->NDArray:
    contribs = cp.repeat(areas*0.33,3)
    flat_particles = tri.ravel()
    return cp.bincount(flat_particles, weights = contribs, minlength = pos.shape[0])

def calc_force_s_vect_gpu(
    ref_pos: NDArray,
    pos: NDArray,
    tri: NDArray,
    active_tri: NDArray,
    areas: NDArray,
    E: float = 10,
    Nu: float = 0.0,
    SE_c: float = 1.0,
    *,
    return_strain_stress: bool = False,
)->tuple:
    Forces = cp.zeros((pos.shape[0],2))
    barrier_count = 0
    n_tri = tri.shape[0]
    E = _as_tri_param_gpu(E, n_tri)
    Nu = _as_tri_param_gpu(Nu, n_tri)
    SE_c = _as_tri_param_gpu(SE_c, n_tri)

    Sq_Nu = Nu*Nu
    E_Nu = E*Nu
    G = E/(2*(1.0+Nu))
    inv_1_minus_Sq_Nu = 1.0/(1.0-Sq_Nu)
    coeff = 0.5*areas
        
    strain_out = None
    stress_out = None
    if return_strain_stress:
        strain_out = cp.full((n_tri, 3), cp.nan, dtype=cp.float64)
        stress_out = cp.full((n_tri, 4), cp.nan, dtype=cp.float64)

    for m in range(3):
        n = (m + 1) % 3
        o = (m + 2) % 3

        n1, n2, n3 = tri[:,m], tri[:,n], tri[:,o]
        xdi, ydi = pos[n1,0], pos[n1,1]
        xdj, ydj = pos[n2,0], pos[n2,1]
        xdk, ydk = pos[n3,0], pos[n3,1]
        xri, yri = ref_pos[n1,0], ref_pos[n1,1]
        xrj, yrj = ref_pos[n2,0], ref_pos[n2,1]
        xrk, yrk = ref_pos[n3,0], ref_pos[n3,1]
        
        rijx = (xrj-xri)
        rijy = (yrj-yri)
        rikx = (xrk-xri)
        riky = (yrk-yri)
        dijx = (xdj-xdi)
        dijy = (ydj-ydi)
        dikx = (xdk-xdi)
        diky = (ydk-ydi)
        
        Temp1 = (dijx*riky-rijy*dikx)
        Temp2 = (dijy*riky-rijy*diky)
        Temp3 = (rijx*dikx-dijx*rikx)
        Temp4 = (rijx*diky-dijy*rikx)
        Temp5 = (rijx*riky-rijy*rikx)
        A = 0.5 * Temp5
        barrier = cp.zeros_like(A)
        barrier_mask = cp.abs(A) < A_EPS
        hit_count = int(cp.count_nonzero(barrier_mask).item())
        if hit_count > 0:
            barrier = cp.where(
                barrier_mask,
                K_BARRIER * cp.where(A >= 0.0, 1.0, -1.0) / (cp.abs(A) + A_EPS),
                0.0,
            )
            barrier_count += hit_count
        
        denom = Temp5*Temp5+1e-20
        inv_d2 = 1.0/denom
        
        EXX = 0.5*(Temp1*Temp1+Temp2*Temp2)*inv_d2-0.5
        EYY = 0.5*(Temp3*Temp3+Temp4*Temp4)*inv_d2-0.5
        EXY = 0.5*(Temp3*Temp1+Temp4*Temp2)*inv_d2
        
        xrkj = (xrk-xrj)
        yrkj = (yrj-yrk)
        
        EXX_X = (yrkj*Temp1)*inv_d2
        EXX_Y = (yrkj*Temp2)*inv_d2
        EYY_X = (xrkj*Temp3)*inv_d2
        EYY_Y = (xrkj*Temp4)*inv_d2
        EXY_X = 0.5*(yrkj*Temp3+xrkj*Temp1)*inv_d2
        EXY_Y = 0.5*(xrkj*Temp2+yrkj*Temp4)*inv_d2
        
        ZXX = E*EXX*inv_1_minus_Sq_Nu+E_Nu*EYY*inv_1_minus_Sq_Nu
        ZYY = E*EYY*inv_1_minus_Sq_Nu+E_Nu*EXX*inv_1_minus_Sq_Nu
        ZXY = G*EXY
        if return_strain_stress and m == 0:
            strain_out[:, 0] = EXX
            strain_out[:, 1] = EYY
            strain_out[:, 2] = EXY
            stress_out[:, 0] = ZXX
            stress_out[:, 1] = ZYY
            stress_out[:, 2] = ZXY
            stress_out[:, 3] = cp.sqrt(cp.maximum(ZXX * ZXX + ZYY * ZYY - ZXX * ZYY + 3.0 * ZXY * ZXY, 0.0))
        
        ZXX_X = E*EXX_X*inv_1_minus_Sq_Nu+E_Nu*EYY_X*inv_1_minus_Sq_Nu
        ZXX_Y = E*EXX_Y*inv_1_minus_Sq_Nu+E_Nu*EYY_Y*inv_1_minus_Sq_Nu
        ZYY_X = E*EYY_X*inv_1_minus_Sq_Nu+E_Nu*EXX_X*inv_1_minus_Sq_Nu
        ZYY_Y = E*EYY_Y*inv_1_minus_Sq_Nu+E_Nu*EXX_Y*inv_1_minus_Sq_Nu
        ZXY_X = G*EXY_X
        ZXY_Y = G*EXY_Y
        
        fx = -coeff*(ZXX_X*EXX+ZXX*EXX_X+ZYY_X*EYY+ZYY*EYY_X+2.0*(ZXY_X*EXY+ZXY*EXY_X))
        fy = -coeff*(ZXX_Y*EXX+ZXX*EXX_Y+ZYY_Y*EYY+ZYY*EYY_Y+2.0*(ZXY_Y*EXY+ZXY*EXY_Y))
        fx += barrier
        fy += barrier

        SED = 0.5*(ZXX*EXX+ZYY*EYY+2.0*(ZXY*EXY))/areas
        
        active_tri[SED > SE_c] = 0

        fx = fx*active_tri
        fy = fy*active_tri
        
        cp.add.at(Forces[:,0],n1,fx)
        cp.add.at(Forces[:,1],n1,fy)

    if return_strain_stress:
        return Forces, barrier_count, strain_out, stress_out
    return Forces, barrier_count

def calc_force_f_vect_gpu(pos:NDArray, vel:NDArray, tri:NDArray, areas:NDArray, rho:float = 10, mu:float = 1.0, K:float = 1.0, A0:float = 0.1)->NDArray:
    Forces = cp.zeros((pos.shape[0],2))
    
    coef = K*(areas-A0)
    coef = cp.minimum(coef, 0)

    for m in range(3):
        n = (m + 1) % 3
        o = (m + 2) % 3

        n1, n2, n3 = tri[:,m], tri[:,n], tri[:,o]
        xdi, ydi = pos[n1,0], pos[n1,1]
        xdj, ydj = pos[n2,0], pos[n2,1]
        xdk, ydk = pos[n3,0], pos[n3,1]
        
        vxi, vyi = vel[n1,0], vel[n1,1]
        vxj, vyj = vel[n2,0], vel[n2,1]
        vxk, vyk = vel[n3,0], vel[n3,1]
        
        rijx = (xdj-xdi)
        rijy = (ydj-ydi)
        rikx = (xdk-xdi)
        riky = (ydk-ydi)
        vijx = (vxj-vxi)
        vijy = (vyj-vyi)
        vikx = (vxk-vxi)
        viky = (vyk-vyi)
        
        # Pressure force
        fxp = -coef*(pos[n2,1]-pos[n3,1])
        fyp = coef*(pos[n2,0]-pos[n3,0])

        # Viscous force
        fxv = -(2*mu/(rijx*riky-rikx*rijy)**2)*(-2*riky*(vijx*riky-vijy*rikx)+2*rijy*(viky*rijx-vikx*rijy)+(rijy-riky)*(vikx*riky-viky*rikx+vijy*rijx-vijx*rijy))
        fyv = -(2*mu/(rijx*riky-rikx*rijy)**2)*(2*rikx*(vijx*riky-vijy*rikx)-2*rijx*(viky*rijx-vikx*rijy)+(rikx-rijx)*(vikx*riky-viky*rikx+vijy*rijx-vijx*rijy))

        
        cp.add.at(Forces[:,0],n1,fxp)
        cp.add.at(Forces[:,1],n1,fyp)

        
    return Forces


def update_gpu(pos:NDArray, vel:NDArray, masses:NDArray, forces:NDArray, fixed_bc:NDArray, vel_bc:NDArray, dt:float = 1e-5, g:float =0, c:float = 0)->tuple[NDArray, NDArray]:

    inv_m = (1.0/masses).reshape(-1,1)
    vel += forces*inv_m*dt-c*inv_m*vel*dt
    vel[:,1] += -g*dt

    if fixed_bc.size>0:
        vel[fixed_bc,0] = 0
        vel[fixed_bc,1] = 0

    if vel_bc.size>0:
        vel_nodes = vel_bc[:,0].astype(cp.int32)
        vel[vel_nodes,0] = vel_bc[:,1]
        vel[vel_nodes,1] = vel_bc[:,2]
    
    pos += vel*dt
    return pos,vel
        
    
    
    

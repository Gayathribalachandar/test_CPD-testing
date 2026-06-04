import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay
from numba import njit, prange
import numba
import os
os.environ["NUMBA_THREADING_LAYER"] = "omp"

# List of cpu functions
A_EPS = 1e-12
K_BARRIER = 1e-6

def delaunay_triangulate(pos:NDArray[np.floating])->NDArray[np.integer]:
    tri = Delaunay(pos).simplices
    return tri

def calc_ele_areas_cpu(pos:NDArray[np.floating], tri:NDArray[np.integer])->NDArray[np.floating]:
    p1 = pos[tri[:,0]]
    p2 = pos[tri[:,1]]
    p3 = pos[tri[:,2]]
    areas = 0.5*np.abs(p1[:,0]*(p2[:,1]-p3[:,1])+p2[:,0]*(p3[:,1]-p1[:,1])+p3[:,0]*(p1[:,1]-p2[:,1]))
    return areas

def calc_particle_areas_cpu(pos:NDArray[np.floating], tri:NDArray[np.integer], areas:NDArray[np.floating])->NDArray[np.floating]:
    contribs = np.repeat(areas*0.33,3)
    flat_particles = tri.ravel()
    return np.bincount(flat_particles, weights = contribs, minlength = pos.shape[0])

def _as_tri_param(param, n_tri, dtype=float):
    arr = np.asarray(param, dtype=dtype)
    if arr.ndim == 0:
        return np.full((n_tri,), arr.item(), dtype=dtype)
    arr = arr.reshape(-1)
    if arr.size == 1:
        return np.full((n_tri,), float(arr[0]), dtype=dtype)
    if arr.size != n_tri:
        raise ValueError(f"Expected parameter size {n_tri}, got {arr.size}")
    return arr.astype(dtype, copy=False)


def calc_force_s_vect_cpu(ref_pos:NDArray[np.floating], pos:NDArray[np.floating], tri:NDArray[np.integer], active_tri:NDArray[np.bool_], areas:NDArray[np.floating], E:float = 10, Nu:float = 0.0, SE_c:float = 1.0 )->NDArray[np.floating]:
    Forces = np.zeros((pos.shape))
    n_tri = tri.shape[0]
    E = _as_tri_param(E, n_tri, dtype=float)
    Nu = _as_tri_param(Nu, n_tri, dtype=float)
    SE_c = _as_tri_param(SE_c, n_tri, dtype=float)

    Sq_Nu = Nu*Nu
    E_Nu = E*Nu
    G = E/(2*(1.0+Nu))
    inv_1_minus_Sq_Nu = 1.0/(1.0-Sq_Nu)
    coeff = 0.5*areas

    for m in range(3):
        n = (m+1)%3
        o = (m+2)%3

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
        barrier = np.zeros_like(A)
        barrier_mask = np.abs(A) < A_EPS
        if np.any(barrier_mask):
            barrier[barrier_mask] = (
                K_BARRIER
                * np.where(A[barrier_mask] >= 0.0, 1.0, -1.0)
                / (np.abs(A[barrier_mask]) + A_EPS)
            )
        
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
        
        np.add.at(Forces[:,0],n1,fx)
        np.add.at(Forces[:,1],n1,fy)

    return Forces



@njit(parallel=True, fastmath=True)
def zero_tls(tls):
    nt, nn = tls.shape[0], tls.shape[1]
    for t in prange(nt):
        for i in range(nn):
            tls[t,i,0] = 0.0
            tls[t,i,1] = 0.0


@njit(parallel=True, fastmath=True)
def compute_tls(ref_pos, pos, tri, active_tri, areas, tls, barrier_tls, E_tri, Nu_tri, SE_c_tri, strain_tri, stress_tri):
    n_tri = tri.shape[0]

    for idx in prange(n_tri):
        if not active_tri[idx]:
            continue

        E = E_tri[idx]
        Nu = Nu_tri[idx]
        SE_c = SE_c_tri[idx]
        Sq_Nu = Nu * Nu
        inv_1_minus_Sq_Nu = 1.0 / (1.0 - Sq_Nu + 1e-20)
        E_Nu = E * Nu
        G = E / (2.0 * (1.0 + Nu + 1e-20))

        tid = numba.get_thread_id()
        F = tls[tid]

        nodes = tri[idx]
        coeff = 0.5 * areas[idx]

        # Unroll the m loop (m=0,1,2) to reduce overhead a bit
        for m in range(3):
            n = (m + 1) % 3
            o = (m + 2) % 3

            n1 = nodes[m]
            n2 = nodes[n]
            n3 = nodes[o]

            xdi = pos[n1, 0]; ydi = pos[n1, 1]
            xdj = pos[n2, 0]; ydj = pos[n2, 1]
            xdk = pos[n3, 0]; ydk = pos[n3, 1]

            xri = ref_pos[n1, 0]; yri = ref_pos[n1, 1]
            xrj = ref_pos[n2, 0]; yrj = ref_pos[n2, 1]
            xrk = ref_pos[n3, 0]; yrk = ref_pos[n3, 1]

            rijx = (xrj - xri); rijy = (yrj - yri)
            rikx = (xrk - xri); riky = (yrk - yri)
            dijx = (xdj - xdi); dijy = (ydj - ydi)
            dikx = (xdk - xdi); diky = (ydk - ydi)

            Temp1 = (dijx * riky - rijy * dikx)
            Temp2 = (dijy * riky - rijy * diky)
            Temp3 = (rijx * dikx - dijx * rikx)
            Temp4 = (rijx * diky - dijy * rikx)
            Temp5 = (rijx * riky - rijy * rikx)
            A = 0.5 * Temp5
            barrier = 0.0
            if abs(A) < A_EPS:
                barrier = K_BARRIER * (1.0 if A >= 0.0 else -1.0) / (abs(A) + A_EPS)
                barrier_tls[tid] += 1

            denom = Temp5 * Temp5 + 1e-20
            inv_d2 = 1.0 / denom

            EXX = 0.5 * (Temp1 * Temp1 + Temp2 * Temp2) * inv_d2 - 0.5
            EYY = 0.5 * (Temp3 * Temp3 + Temp4 * Temp4) * inv_d2 - 0.5
            EXY = 0.5 * (Temp3 * Temp1 + Temp4 * Temp2) * inv_d2

            xrkj = (xrk - xrj)
            yrkj = (yrj - yrk)

            EXX_X = (yrkj * Temp1) * inv_d2
            EXX_Y = (yrkj * Temp2) * inv_d2
            EYY_X = (xrkj * Temp3) * inv_d2
            EYY_Y = (xrkj * Temp4) * inv_d2
            EXY_X = 0.5 * (yrkj * Temp3 + xrkj * Temp1) * inv_d2
            EXY_Y = 0.5 * (xrkj * Temp2 + yrkj * Temp4) * inv_d2

            ZXX = (E * EXX + E_Nu * EYY) * inv_1_minus_Sq_Nu
            ZYY = (E * EYY + E_Nu * EXX) * inv_1_minus_Sq_Nu
            ZXY = G * EXY

            if m == 0:
                strain_tri[idx, 0] = EXX
                strain_tri[idx, 1] = EYY
                strain_tri[idx, 2] = EXY

                stress_tri[idx, 0] = ZXX
                stress_tri[idx, 1] = ZYY
                stress_tri[idx, 2] = ZXY
                stress_tri[idx, 3] = np.sqrt(max(ZXX * ZXX + ZYY * ZYY - ZXX * ZYY + 3.0 * ZXY * ZXY, 0.0))


            ZXX_X = (E * EXX_X + E_Nu * EYY_X) * inv_1_minus_Sq_Nu
            ZXX_Y = (E * EXX_Y + E_Nu * EYY_Y) * inv_1_minus_Sq_Nu
            ZYY_X = (E * EYY_X + E_Nu * EXX_X) * inv_1_minus_Sq_Nu
            ZYY_Y = (E * EYY_Y + E_Nu * EXX_Y) * inv_1_minus_Sq_Nu
            ZXY_X = G * EXY_X
            ZXY_Y = G * EXY_Y

            fx = -coeff * (ZXX_X * EXX + ZXX * EXX_X +
                           ZYY_X * EYY + ZYY * EYY_X +
                           2.0 * (ZXY_X * EXY + ZXY * EXY_X))
            fy = -coeff * (ZXX_Y * EXX + ZXX * EXX_Y +
                           ZYY_Y * EYY + ZYY * EYY_Y +
                           2.0 * (ZXY_Y * EXY + ZXY * EXY_Y))
            fx += barrier
            fy += barrier

            F[n1, 0] += fx
            F[n1, 1] += fy

            # Damage / failure 
            SED = 0.5 * (ZXX * EXX + ZYY * EYY + 2.0 * (ZXY * EXY)) / areas[idx]
            if SED > SE_c:
                active_tri[idx] = False


@njit(parallel=True, fastmath=True)
def reduce_tls(tls, forces):
    nt, nn = tls.shape[0], tls.shape[1]
    for i in prange(nn):
        s0 = 0.0
        s1 = 0.0
        for t in range(nt):
            s0 += tls[t,i,0]
            s1 += tls[t,i,1]
        forces[i,0] = s0
        forces[i,1] = s1



@njit(cache=True, fastmath=True)
def calc_force_s_numba_cpu(ref_pos:NDArray[np.floating], pos:NDArray[np.floating], tri:NDArray[np.integer], active_tri:NDArray[np.bool_], areas:NDArray[np.floating], E:float = 10, Nu:float = 0.0, SE_c:float = 1.0)->NDArray[np.floating]:
    Forces = np.zeros((pos.shape[0],2))

    Sq_Nu = Nu*Nu
    E_Nu = E*Nu
    G = E/(2*(1.0+Nu))
    inv_1_minus_Sq_Nu = 1.0/(1.0-Sq_Nu)
    
    for idx,nodes in enumerate(tri):
        if active_tri[idx] == 1:
            coeff = areas[idx]*0.5
            for m in range(3):
                n = (m+1)%3
                o = (m+2)%3
                
                n1, n2, n3 = nodes[m], nodes[n], nodes[o]
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
                barrier = 0.0
                if abs(A) < A_EPS:
                    barrier = K_BARRIER * (1.0 if A >= 0.0 else -1.0) / (abs(A) + A_EPS)
                
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
                
                ZXX_X = E*EXX_X*inv_1_minus_Sq_Nu+E_Nu*EYY_X*inv_1_minus_Sq_Nu
                ZXX_Y = E*EXX_Y*inv_1_minus_Sq_Nu+E_Nu*EYY_Y*inv_1_minus_Sq_Nu
                ZYY_X = E*EYY_X*inv_1_minus_Sq_Nu+E_Nu*EXX_X*inv_1_minus_Sq_Nu
                ZYY_Y = E*EYY_Y*inv_1_minus_Sq_Nu+E_Nu*EXX_Y*inv_1_minus_Sq_Nu
                ZXY_X = G*EXY_X
                ZXY_Y = G*EXY_Y
                
                Forces[n1,0] += -coeff*(ZXX_X*EXX+ZXX*EXX_X+ZYY_X*EYY+ZYY*EYY_X+2.0*(ZXY_X*EXY+ZXY*EXY_X)) + barrier
                Forces[n1,1] += -coeff*(ZXX_Y*EXX+ZXX*EXX_Y+ZYY_Y*EYY+ZYY*EYY_Y+2.0*(ZXY_Y*EXY+ZXY*EXY_Y)) + barrier

                SED = 0.5*(ZXX*EXX+ZYY*EYY+2.0*(ZXY*EXY))/areas[idx]

                if (SED>SE_c):
                    active_tri[idx] = 0

        else:
            for m in range(3):
                n = (m+1)%3
                o = (m+2)%3
                
                n1, n2, n3 = nodes[m], nodes[n], nodes[o]    

                Forces[n1,0] += 0
                Forces[n1,1] += 0

    return Forces

def calc_force_f_vect_cpu(pos:NDArray[np.floating], vel:NDArray[np.floating], tri:NDArray[np.integer], areas:NDArray[np.floating], rho:float = 10, mu:float = 1.0, K:float = 1.0, A0:float = 0.1)->NDArray[np.floating]:
    Forces = np.zeros((pos.shape[0],2))
    
    coef = K*(areas-A0)
    coef = np.minimum(coef, 0)

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

        
        np.add.at(Forces[:,0],n1,fxp)
        np.add.at(Forces[:,1],n1,fyp)

        
    return Forces

@njit(cache=True, fastmath=True)
def calc_force_f_numba_cpu(pos:NDArray[np.floating], vel:NDArray[np.floating], tri:NDArray[np.integer], areas:NDArray[np.floating], rho:float = 10, mu:float = 1.0, K:float = 1.0, A0:float = 0.1)->NDArray[np.floating]:
    Forces = np.zeros((pos.shape[0],2))
    
    for idx,nodes in enumerate(tri):
        coef = K*(areas[idx]-A0)
        for m in range(3):
            n = (m+1)%3
            o = (m+2)%3
            n1, n2, n3 = nodes[m], nodes[n], nodes[o]
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
            
            if coef < 0:
                # Pressure force
                fxp = -coef*(pos[n2,1]-pos[n3,1])
                fyp = coef*(pos[n2,0]-pos[n3,0])
                
            else:
                fxp = 0
                fyp = 0

            # Viscous force
            fxv = -(2*mu/(rijx*riky-rikx*rijy)**2)*(-2*riky*(vijx*riky-vijy*rikx)+2*rijy*(viky*rijx-vikx*rijy)+(rijy-riky)*(vikx*riky-viky*rikx+vijy*rijx-vijx*rijy))
            fyv = -(2*mu/(rijx*riky-rikx*rijy)**2)*(2*rikx*(vijx*riky-vijy*rikx)-2*rijx*(viky*rijx-vikx*rijy)+(rikx-rijx)*(vikx*riky-viky*rikx+vijy*rijx-vijx*rijy))

            
            Forces[n1,0] += fxp
            Forces[n1,1] += fyp
        
    return Forces

def update_cpu(pos:NDArray[np.floating], vel:NDArray[np.floating], masses:NDArray[np.floating], forces:NDArray[np.floating], fixed_bc:NDArray[np.integer], vel_bc:NDArray[np.floating], dt:float = 1e-5, g:float = 0, c:float = 0)->tuple[:NDArray[np.floating], :NDArray[np.floating]]:
    
    #force_nodes = force_bc[:, 0].astype(int)
    vel_nodes = vel_bc[:, 0].astype(int)
    
    #if force_bc.size>0:
    #    forces[force_nodes,0] = force_bc[:,1]
    #    forces[force_nodes,1] = force_bc[:,2]

    inv_m = (1.0/masses).reshape(-1,1)
    vel += forces*inv_m*dt-c*inv_m*vel*dt
    vel[:,1] += -g*dt

    if fixed_bc.size>0:
        vel[fixed_bc,0] = 0
        vel[fixed_bc,1] = 0
        
    if vel_bc.size>0:
        vel[vel_nodes,0] = vel_bc[:,1]
        vel[vel_nodes,1] = vel_bc[:,2]
    
    pos += vel*dt
    return pos,vel

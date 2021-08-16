#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import xarray as xr
import pandas as pd
import numba


# # Continuous Equation
# 
# $$
#     \begin{align}
#         \frac{\partial h}{\partial t}+\frac{\partial q_x}{\partial x} +\frac{\partial q_y}{\partial y} = 0
#     \end{align}
# $$

# In[ ]:


@numba.jit(nopython=True, parallel=True)
def conEq(dep, qx, qy, dzb, dt, dx, dy, ibx, hmin, hbuf, hdown, periodic=True):
#     Qind = range(154,159)
    
    imax, jmax = len(dep), len(dep[0])
    depn = np.zeros_like(dep, dtype=np.float64)
    fluxx = np.zeros((imax+1, jmax), dtype=np.float64)
    fluxy = np.zeros((imax, jmax+1), dtype=np.float64)
    modflux = np.full( (imax, jmax), False)
    
    gravity = float( 9.8 )
    
    f = lambda Qp, Qm : Qm if Qp >= 0.0 and Qm >= 0.0 else (Qp if Qp <= 0.0 and Qm <= 0.0 else 0.5*Qp+0.5*Qm )
    
    def flux(Qp, Qm, depp, depm, zbp, zbm, ib, delta) : 
        r = f(Qp, Qm)
#         if ( (depm + zbm) < zbp - ib*delta) and (depp <= hbuf) : r = 0.0
#         if ( (depp + zbp) < zbm + ib*delta) and (depm <= hbuf) : r = 0.0
        if ( (depm + zbm) <= zbp + hbuf - ib*delta) and (depp <= hbuf) : r = 0.0
        if ( (depp + zbp) <= zbm + hbuf + ib*delta) and (depm <= hbuf) : r = 0.0
            
        return r
        
    for i in numba.prange( imax ):
        for j in range( jmax ):
            c, xm = (i,j), (i-1,j)
            fluxx[c] = flux(qx[c], qx[xm], dep[c], dep[xm], dzb[c], dzb[xm], ibx, dx)
            
    if periodic :
# boundary : periodic
        fluxx[-1,:] = fluxx[0,:] 
    else:
        for j in numba.prange( jmax ): fluxx[-1,j] = fluxx[-2,j] # qx[-1,j] # if qx[-1,j] > 0.0 else qx[,j]
# normal            
#         for j in numba.prange( jmax ): fluxx[-1,j] = qx[-1,j] if qx[-1,j] > 0.0 else 0.0
        
    for i in numba.prange( imax ):
        for j in range( 1, jmax ):
            c, ym = (i,j), (i,j-1)
            fluxy[c] = flux(qy[c], qy[ym], dep[c], dep[ym], dzb[c], dzb[ym], 0.0, dy)
            
# wall boundary 
#     fluxy[:,0] = 0.0 
#     fluxy[:,-1] = 0.0 
    
    for i in numba.prange( imax ):
        fluxy[i,-1] = qy[i,-1] if qy[i,-1] > 0.0 else 0.0
        fluxy[i, 0] = qy[i, 0] if qy[i, 0] < 0.0 else 0.0    
    
    nis = 0 if periodic else 1
# limiter --------------------------------------------------------------
# 水深が負になる際に質量保存を満たすためにフラックスを修正する
    for i in range(nis, imax):
        for j in range(jmax):
            if dep[c] > hmin :
                c, xp, yp = (i, j), (i+1, j), (i, j+1)
                fxp = fluxx[xp] if fluxx[xp] > 0.0 else 0.0
                fxm = -fluxx[c] if fluxx[c] < 0.0 else 0.0
                fyp = fluxy[yp] if fluxy[yp] > 0.0 else 0.0
                fym = -fluxy[c] if fluxy[c] < 0.0 else 0.0
                V =  dep[c]*dx*dy - hmin*dx*dy 
                Vq = ( fxp*dy + fxm*dy + fyp*dx + fym*dx )*dt
                if V <  Vq:
                    alfa = V / Vq - 0.001
                    if fluxx[xp] > 0.0 : fluxx[xp] *= alfa 
                    if fluxx[c]  < 0.0 : fluxx[c]  *= alfa
                    if fluxy[yp] > 0.0 : fluxy[yp] *= alfa
                    if fluxy[c]  < 0.0 : fluxy[c]  *= alfa
                        
                    modflux[c] = True
# ------------------------------------------------------------------------
    n = 0
    for i in numba.prange(nis, imax):
        for j in range(jmax):
            c, xp, yp = (i, j), (i+1, j), (i, j+1)
            depn[c] = dep[c] - dt*(fluxx[xp] - fluxx[c])/dx - dt*(fluxy[yp] - fluxy[c])/dy
            if depn[c] < hmin : 
                n += 1
#                 print('dep-error')
#                 print( modflux[c], depn[c], fluxx[xp], fluxx[c], fluxy[yp], fluxy[c] )
                fxp = fluxx[xp] if fluxx[xp] > 0.0 else 0.0
                fxm = -fluxx[c] if fluxx[c] < 0.0 else 0.0
                fyp = fluxy[yp] if fluxy[yp] > 0.0 else 0.0
                fym = -fluxy[c] if fluxy[c] < 0.0 else 0.0
                V =  dep[c]*dx*dy - hmin*dx*dy 
                Vq = ( fxp*dy + fxm*dy + fyp*dx + fym*dx )*dt
#                 print(V,Vq)
                
                depn[c] = hmin
                
# upstream boundary                
#     if periodic == False: depn[0][:] = depn[1][:]
    if periodic == False: 
#         depn[0][:] = dep[0][:]
#         for j in Qind : depn[0,j] = depn[1,j]  
            
        depn[0,:] = depn[1,:]  
    
# downstream boundary 
#     depn[-1][:] = hdown
    depn[-1][:] = depn[-2][:]
    
    return depn, n


# # Momentum Equation
# 
# $$
#     \begin{align}
#         \frac{\partial q_x}{\partial t}+\frac{\partial u q_x}{\partial x}+\frac{\partial v q_x}{\partial y}+gh\frac{\partial H}{\partial x}+\frac{\tau_{0x}}{\rho} 
#         - \nu_t h \left(\frac{\partial^2 u}{\partial x^2}+\frac{\partial^2 u}{\partial y^2} \right)= 0 \\
#         \frac{\partial q_y}{\partial t}+\frac{\partial u q_y}{\partial x}+\frac{\partial v q_y}{\partial y}+gh\frac{\partial H}{\partial y}+\frac{\tau_{0y}}{\rho}- \nu_t h \left(\frac{\partial^2 v}{\partial x^2}+\frac{\partial^2 v}{\partial y^2} \right)
#         = 0
#     \end{align}
# $$

# In[ ]:


@numba.jit(nopython=True, parallel=True)
def momentEq(dep, qx, qy, depn, dzb, dt, dx, dy, ibx, hmin, hbuf, direction, Qup, cManning, periodic=True):
    #direction = 1:x, 2:y
    gravity = float( 9.8 )
#     Qind = range(154,159)

    q = qx.copy() if direction == 1 else qy.copy()
    u, v = qx/dep, qy/dep
    Vdir = q/dep  
        
    imax, jmax = len(q), len(q[0])
    
    qn = np.zeros_like(q, dtype=np.float64)
    fluxx = np.zeros((imax+1, jmax), dtype=np.float64)
    fluxy = np.zeros((imax, jmax+1), dtype=np.float64)
            
    f = lambda vp,vm,qp,qm : vm*qm if vp >= 0.0 and vm >= 0.0 else         (vp*qp if vp <= 0.0 and vm <= 0.0 else (0.5*vp+0.5*vm)*(0.5*qp+0.5*qm) )
        
    def flux1(vp, vm, qp, qm, depp, depm, zbp, zbm, ib, delta) : 
        r = f(vp,vm,qp,qm)
#         if ( (depm + zbm) < zbp - ib*delta) and (depp <= hbuf) : r = 0.0
#         if ( (depp + zbp) < zbm + ib*delta) and (depm <= hbuf) : r = 0.0
        if ( (depm + zbm) <= zbp + hbuf - ib*delta) and (depp <= hbuf) : r = 0.0
        if ( (depp + zbp) <= zbm + hbuf + ib*delta) and (depm <= hbuf) : r = 0.0
            
        return r
    
    for i in numba.prange( imax ):
        for j in range( jmax ):
            c, xm = (i,j), (i-1,j)
            fluxx[c] = flux1( u[c], u[xm], q[c], q[xm], dep[c], dep[xm], dzb[c], dzb[xm], ibx, dx )
            
# boundary : periodic
    if periodic :
        fluxx[-1,:] = fluxx[0,:] 
    else:
        fluxx[0,:]  = -9999
        for j in numba.prange( jmax ):
            fluxx[-1,j] = u[-1,j]*q[-1,j] if u[-1,j] > 0.0 else 0.0 
    
    for i in numba.prange( imax ):
        for j in range( 1, jmax ):
            c, ym = (i,j), (i,j-1)
            fluxy[c] = flux1( v[c], v[ym], q[c], q[ym], dep[c], dep[ym], dzb[c], dzb[ym], 0.0, dy )
            
# wall boundary 
#     fluxy[:,0]  = -fluxy[:,1]
#     fluxy[:,-1] = -fluxy[:,-2]

    for i in numba.prange( imax ):
        fluxy[i,-1] = v[i,-1]*q[i,-1] if v[i,-1] > 0.0 else 0.0
        fluxy[i, 0] = v[i, 0]*q[i, 0] if v[i,0 ] < 0.0 else 0.0
    
    nis = 0 if periodic else 1
    
    for i in numba.prange(nis, imax):
        for j in range(jmax):    
            c = (i, j)
            if periodic :
                xp = (0, j) if i == imax-1 else (i+1, j)                
            else:
                xp = (i+1, j)                
            xm = (i-1, j)
            yp = (i, j+1)
            ym = (i, j-1)

            if depn[c] <= hbuf :
                qn[c] = 0.0
            else:
            # pressure & gravity term
                if direction == 2 and ((j == 0) or (j == jmax-1)) : 
                    dHdx = 0.0
                elif direction == 1 and i == imax-1 :
#                     dHdx = 0.0
                    dHdx =((depn[i,j]+dzb[i,j]) - (depn[i-1,j]+dzb[i-1,j]))/dx
                else :
                    if direction == 1 :
                        dp, dm, delta  = xp, xm, dx
                        dib = ibx
                    else :
                        dp, dm, delta = yp, ym, dy
                        dib = 0.0
                        
                    Vc, Vp, Vm = q[c]/dep[c], q[dp]/dep[dp], q[dm]/dep[dm]
                    Hc, Hp, Hm = depn[c]+dzb[c], depn[dp]+dzb[dp], depn[dm]+dzb[dm]

#                     if(Hc < dzb[dp] - dib*delta) and depn[dp] <= hbuf :
                    if(Hc <= dzb[dp] + hbuf - dib*delta) and depn[dp] <= hbuf :
#                         if(Hc < dzb[dm] + dib*delta) and depn[dm] <= hbuf :
                        if(Hc <= dzb[dm] + hbuf + dib*delta) and depn[dm] <= hbuf :
                            dHdx = 0.0
                        else:
                            dHdx = (Hc-Hm)/delta-dib
#                     elif(Hc < dzb[dm] + dib*delta) and depn[dm] <= hbuf :
                    elif(Hc <= dzb[dm] + hbuf + dib*delta) and depn[dm] <= hbuf :
                        dHdx = (Hp-Hc)/delta-dib
                    else :
                        if Vc > 0.0 and Vp > 0.0 and Vm > 0.0: 
                            Cr1, Cr2 = 0.5*(abs(Vc)+abs(Vp))*dt/delta, 0.5*(abs(Vc)+abs(Vm))*dt/delta
                            dHdx1, dHdx2 = (Hp-Hc)/delta-dib, (Hc-Hm)/delta-dib
                        elif Vc < 0.0 and Vp < 0.0 and Vm < 0.0:
                            Cr1, Cr2 = 0.5*(abs(Vc)+abs(Vm))*dt/delta, 0.5*(abs(Vc)+abs(Vp))*dt/delta
                            dHdx1, dHdx2 = (Hc-Hm)/delta-dib, (Hp-Hc)/delta-dib          
                        else:
                            Cr1 = Cr2 = 0.5*(abs(0.5*(Vc+Vp))+abs(0.5*(Vc+Vm)))*dt/delta
                            dHdx1 = dHdx2 = (0.5*(Hc+Hp) - 0.5*(Hc+Hm)) / delta - dib
            
                        w1, w2 = 1-Cr1**0.5, Cr2**0.5
                        dHdx = w1 * dHdx1 + w2 * dHdx2   
                
# viscous sublayer
#                 Cf = 1.0/(1.0/0.4*np.log(11.0*dep[c]/2.0/di))**2
                Cf = gravity*cManning**2.0/dep[c]**(1.0/3.0) 
                Vnorm = np.sqrt(u[c]**2.0+v[c]**2.0) 
                Vis = Cf * Vnorm * u[c] if direction == 1 else  Cf * Vnorm * v[c]
            
# turbulence
#                kenergy = 2.07*Cf*Vnorm**2
                nut = 0.4/6.0*dep[c]*np.sqrt(Cf)*np.abs(Vnorm)

# side boundary : non-slip condition
                if (i == imax-1) or (j == 0) or (i == jmax-1): 
                    turb = 0.0
                else :
#                     turb = nut * ( Vdir[xp] - 2.0*Vdir[c] + Vdir[xm] )/ dx**2 \
#                          + nut * ( Vdir[yp] - 2.0*Vdir[c] + Vdir[ym] )/ dy**2
                    
                    turb = nut * ( q[xp] - 2.0*q[c] + q[xm] )/ dx**2                          + nut * ( q[yp] - 2.0*q[c] + q[ym] )/ dy**2
               #tmp         
#                 turb = 0.0
#                 sourcet = Vis - dep[c] * turb
                sourcet = Vis - turb
                
                qn[c] = q[c] - dt * ( fluxx[xp] - fluxx[c] ) / dx                              - dt * ( fluxy[yp] - fluxy[c] ) / dy                              - dt * gravity * depn[c] * dHdx                              - dt * sourcet
                    
                
    if periodic == False :
# upstream boundary
        if direction == 2 : 
            qn[0,:] = 0.0
        else :
#             updep = depn[0,:].copy()
#             for j in range(jmax):
#                 updep[j] = 0.0 if updep[j] <= hmin else updep[j]
                
#             alpha = Qup / dy / np.sum( updep[:]**(5/3) )
#             qn[0,:] = alpha * updep[:]**(5/3)
        
#             qn[0,:] = 0.0
#             qn[0,117] = 0.001

            updep = np.zeros_like(depn[0,:])
            for j in range(jmax):
                if depn[0,j] > hmin : updep[j] = depn[0,j]
                    
            alpha = Qup / dy / np.sum( updep[:]**(5/3) )
            qn[0,:] = alpha * updep[:]**(5/3)
            
# downstream boundary
#     qn[-1,:] = qn[-2,:]
        
    return qn


# In[ ]:


@numba.jit(nopython=True, parallel=False)
def simulation(dep, qx, qy, dzb, dt, dx, dy, ibx, hmin, hbuf, Qup, hdown, cManning):
    depn, count = conEq(dep, qx, qy, dzb, dt, dx, dy, ibx, hmin, hbuf, hdown, False)
    qxn = momentEq(dep, qx, qy, depn, dzb, dt, dx, dy, ibx, hmin, hbuf, 1, Qup, cManning, False)
    qyn = momentEq(dep, qx, qy, depn, dzb, dt, dx, dy, ibx, hmin, hbuf, 2, Qup, cManning, False)
#     CFL = ((np.abs(qxn/depn) + np.sqrt(9.8*depn))/dx + ( np.abs(qyn/depn) + np.sqrt(9.8*depn) )/dy)*dt
    CFL = ((np.abs(qxn/depn))/dx + ( np.abs(qyn/depn))/dy)*dt
    CFLmax = np.max(CFL)
    return depn, qxn, qyn, CFLmax, count


# # main 

# In[ ]:


ds = xr.open_dataset('zb.nc')

hmin = float(10.0**(-5)) 
hbuf = float(10.0**(-2)) 

cManning = 0.025

dx, dy = float(2.5), float(2.5)
dtini = float(0.05)

nxmax, nymax = ds.dims['x'], ds.dims['y']
qx = np.zeros((nxmax,nymax), dtype=np.float64)
qy = np.zeros_like(qx, dtype=np.float64)
dep = np.full_like(qx, hmin, dtype=np.float64)
zb = np.zeros_like(qx, dtype=np.float64)
zb[:,:] = ds['elevation'].values[:,:]

# initial condition
WL = 27.0
dep0 = WL - zb[0,:]
dep0 = np.where(dep0<hmin, hmin, dep0)
dep[0,:] = dep0
dep[1,:] = dep0


# In[ ]:


# %%time
Qt = float(100)
dic = ds.attrs

# CFL = float(0.4)

t = float(0)
dt = 0.025
dtout= float(3600.0)
tmax = float(5.0*3600.+0.1)
nout = 0

while tmax >= t :
    t += dt
    dep, qx, qy, CFLmax, count = simulation(dep, qx, qy, zb, dt, dx, dy, float(0), hmin, hbuf, Qt, float(0), cManning)
    
# update dt
#     dt = np.round( dt * CFL/CFLmax, 5) 
    
    if t >= nout*dtout :
        print(t, dt, CFLmax, count)
        dic['total_second'] = round(t, 2)
        dss = xr.Dataset({'depth': (['x','y'], dep), 'u': (['x','y'], qx/dep), 'v': (['x','y'], qy/dep)
                          , 'elevation': (['x','y'], zb) }
                          , coords={'xc': (('x', 'y'), ds['xc']), 'yc': (('x', 'y'), ds['yc'])}
                          , attrs=dic )
        
        out = dss.to_netcdf('out' + str(nout).zfill(8) + '.nc')
        out = dss.close()
        del out
        nout += 1 


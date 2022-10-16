#Dario Maddaloni
import numpy as np
import time

def g(x):
    return x**2/10

def f(x):
    return (x-1)**3
    
    
#1.1
def gs_step_1d(uh, fh):
    N = len(uh) - 1
    h = 4/N
    pseudo_residual=0
    for i in range(1,N):
        olduh_i=uh[i]
        uh[i]=(h**2 *fh[i]+uh[i-1]+uh[i+1])/(2+h**2) #note that python perform firstly the exponentiation, then the multiplication and at the end the sum
        pseudo_residual=max(abs(olduh_i-uh[i]), pseudo_residual)
    return pseudo_residual    

#1.3
def restriction(vh_fine):
    vh_coarse=[vh_fine[0]]
    for i in range(1,(len(vh_fine)-1)//2):
        vh_coarse.append((vh_fine[2*i-1]+2*vh_fine[2*i]+vh_fine[2*i+1])/4)
    vh_coarse.append(vh_fine[-1])
    return np.array(vh_coarse)

def prolongation(vh_coarse):
    vh_fine=[]
    for i in range(len(vh_coarse) -1):
        vh_fine.append(vh_coarse[i])
        vh_fine.append((vh_coarse[i]+vh_coarse[i+1])/2)
    vh_fine.append(vh_coarse[-1])
    return np.array(vh_fine)
 
 #1.4
 def v_cycle_step_1d(uh, fh, alpha1, alpha2):
    N=len(uh)-1
    h=4/N
    
    val=1/h**2
    Ah= np.diag([1]+[2*val +1 for _ in range(N-1)]+[1])+np.diag([0]+[-1*val for _ in range(N-1)],1)+np.diag([-1*val for _ in range(N-1)]+[0],-1)

    if N==1:
        uh[:]=np.linalg.solve(Ah,fh)
        return 0
    else:
        '1'
        for _ in range(alpha1):
            gs_step_1d(uh, fh)
            
        '2'
        rh=fh-Ah@uh

        r2h=restriction(rh)
        
        '3'
        e2h=np.zeros((N //2)+1)
        
        '4'
        v_cycle_step_1d(e2h, r2h, alpha1, alpha2)
        
        '5'
        eh=prolongation(e2h)
        uh+=eh
        '''uh[:]=[uh[i]+eh[i] for i in range(len(uh))]'''
        '6'
        for _ in range(alpha2-1):
            gs_step_1d(uh, fh)
            
        var_controllo=uh.copy()
        return gs_step_1d(uh, fh)


#1.6
def natural_restriction(vh_fine):
    vh_coarse=[]
    for i in range((len(vh_fine))//2 +1):
        vh_coarse.append(vh_fine[2*i])
    return np.array(vh_coarse)

def full_mg_1d(uh, fh, alpha1, alpha2, nu):
    N=len(uh)-1
    h=4/N
    
    if N==1:
        val=1/h**2
        #Ah is the identity matrix
        Ah= np.diag([1]+[2*val +1 for _ in range(N-1)]+[1])+np.diag([0]+[-1*val for _ in range(N-1)],1)+np.diag([-1*val for _ in range(N-1)]+[0],-1)
        uh[:]=np.linalg.solve(Ah,fh)
        return 0
    else:
        f2h=natural_restriction(fh)
        u2h=np.zeros_like(f2h)
        full_mg_1d(u2h, f2h , alpha1, alpha2, nu)
        uh[:]=prolongation(u2h)
        for _ in range(nu-1):
            v_cycle_step_1d(uh, fh, alpha1, alpha2)
        return v_cycle_step_1d(uh, fh, alpha1, alpha2)


def f2(x1, x2):
    return x1**2-x2**2

def g2(x1, x2):
    return (x1**2+x2**2)/10

def gs_step_2d(uh, fh):
    N = len(uh)-1
    h = 4/N
    pseudo_residual=0
    for i in range(1,N):
        for j in range(1,N):
            old_uh_ij=uh[i][j]
            uh[i,j] = (h**2 *fh[i,j] + uh[i-1,j] + uh[i+1,j] + uh[i,j-1] + uh[i,j+1]) / (4+h**2)
            pseudo_residual=max(abs(old_uh_ij-uh[i][j]), pseudo_residual)
    return pseudo_residual

def restriction_2d(vh_fine):
    N_coarse=(len(vh_fine)-1)//2
    boundary_condition=np.array([vh_fine[i,0] for i in range(0,len(vh_fine),2)])
    vh_coarse=np.zeros((N_coarse+1,N_coarse+1))
    vh_coarse[:,0]=boundary_condition
    vh_coarse[:,N_coarse]=boundary_condition
    vh_coarse[0,:]=boundary_condition
    vh_coarse[N_coarse,:]=boundary_condition
    for i in range(1,N_coarse):
        for j in range(1,N_coarse):
            vh_coarse[i,j]=(vh_fine[2*i-1,2*j-1] + vh_fine[2*i-1,2*j+1] + vh_fine[2*i+1,2*j-1] + vh_fine[2*i+1,2*j+1] + 2*(vh_fine[2*i,2*j-1] + vh_fine[2*i, 2*j+1] + vh_fine[2*i-1,2*j] + vh_fine[2*i+1,2*j]) + 4*vh_fine[2*i,2*j])/16
    return vh_coarse
    

def prolongation_2d_boundaried(vh_coarse):
    N_fine=(len(vh_coarse)-1)*2
    
    vh_fine=np.zeros((N_fine+1,N_fine+1))
    
    for i in range(N_fine//2):
        for j in range(N_fine//2):
            vh_fine[2*i,2*j] = vh_coarse[i,j]
            vh_fine[2*i+1,2*j] = (vh_coarse[i,j] + vh_coarse[i+1,j]) /2
            vh_fine[2*i,2*j+1] = (vh_coarse[i,j] + vh_coarse[i,j+1]) /2
            vh_fine[2*i+1, 2*j+1] = (vh_coarse[i,j] + vh_coarse[i+1,j] + vh_coarse[i,j+1] + vh_coarse[i+1,j+1]) /4
    
    'Boundaries condition'
    vh_fine[:,0]=np.array(g2(np.linspace(-2,2,N_fine+1),-2))
    vh_fine[:,N_fine]=np.array(g2(np.linspace(-2,2,N_fine+1),2))
    vh_fine[0,:]=np.array(g2(-2,np.linspace(-2,2,N_fine+1)))
    vh_fine[N_fine,:]=np.array(g2(2,np.linspace(-2,2,N_fine+1)))
    
    return vh_fine
    
def prolongation_2d_homogeneous(vh_coarse):
    N_coarse=len(vh_coarse)-1
    N_fine=(N_coarse)*2
    vh_fine=np.zeros((N_fine+1,N_fine+1))
    for i in range(N_coarse):
        for j in range(N_coarse):
            vh_fine[2*i,2*j] = vh_coarse[i,j]
            vh_fine[2*i+1,2*j] = (vh_coarse[i,j] + vh_coarse[i+1,j]) /2
            vh_fine[2*i,2*j+1] = (vh_coarse[i,j] + vh_coarse[i,j+1]) /2
            vh_fine[2*i+1, 2*j+1] = (vh_coarse[i,j] + vh_coarse[i+1,j] + vh_coarse[i,j+1] + vh_coarse[i+1,j+1]) /4    
    return vh_fine

def fh_minus_Ah_dot_uh(uh, fh):
    N=len(uh)-1
    h=4/N
    rh=np.zeros((N+1,N+1))
    for i in range(1,N):
        for j in range(1,N):
            rh[i,j] = fh[i,j]-((4*uh[i,j] - uh[i-1,j] - uh[i+1,j] - uh[i,j-1] - uh[i,j+1]) / (h**2) + uh[i,j])
    return rh
    
def v_cycle_step_2d(uh, fh, alpha1, alpha2):
    N=len(uh)-1
    h=4/N
    if N==1:
        uh=np.array([[4/5,4/5],[4/5,4/5]])
        return 0
    else:
        for _ in range(alpha1):
            gs_step_2d(uh, fh)
        rh=fh_minus_Ah_dot_uh(uh, fh)
        r2h=restriction_2d(rh)
        e2h=np.zeros(((N //2)+1,(N //2)+1))
        v_cycle_step_2d(e2h, r2h, alpha1, alpha2)
        uh+=prolongation_2d_homogeneous(e2h)
        for _ in range(alpha2-1):
            gs_step_2d(uh, fh)
        return gs_step_2d(uh, fh)
        


#2.1
def full_mg_2d(uh, fh, alpha1, alpha2, nu):
    N=len(uh)-1
    h=4/N
    
    if N==1:
        uh[:]=(np.identity(2)).dot(uh)
        return 0
    else:
        f2h=restriction_2d(fh)
        u2h=np.zeros_like(f2h)
        full_mg_2d(u2h, f2h, alpha1, alpha2, nu)
        ###############ATTENZIONE###############ho usato una prolongation homoheneous anche se dovrebbe essere boundaried
        uh[:]=prolongation_2d_boundaried(u2h)
        for _ in range(nu-1):
            v_cycle_step_2d(uh, fh, alpha1, alpha2)
        return v_cycle_step_2d(uh, fh, alpha1, alpha2)

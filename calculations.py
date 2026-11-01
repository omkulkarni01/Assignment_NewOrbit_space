"""Reproducible assignment calculations. SI units; Python 3, scipy, numpy, CoolProp, matplotlib."""
import json, math
import numpy as np
from scipy.optimize import brentq
import CoolProp, CoolProp.CoolProp as CP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
out=Path(__file__).parent
R=8.31446261815324; e=1.602176634e-19; NA=6.02214076e23; g=9.80665
Tstd=273.15; pstd=101325.; V=.0011
mol_sccm=pstd*1e-6/(60*R*Tstd)
def pressure(fluid,T,rho):
 if fluid=='Xenon': return CP.PropsSI('P','T',T,'Dmass',rho,fluid)
 s=CP.AbstractState('HEOS','Nitrogen&Oxygen');s.set_mole_fractions([.5,.5]);s.specify_phase(CP.iphase_gas);s.update(CP.DmassT_INPUTS,rho,T);return s.p()
res={'CoolProp':CoolProp.__version__,'beam':[],'storage':{},'ideal_beam':[]}
for I in [.100,.099,.060]:
 for name,M,F in [('Xe',.131293,I*math.sqrt(2*.131293/NA*2000/e)),('N2/O2',.0300061,I/2*sum(math.sqrt(2*m/NA*2000/e) for m in [.0280134,.0319988]))]:
  res['ideal_beam'].append(dict(gas=name,I_A=I,F_mN=1000*F,Isp_s=F/(3*mol_sccm*M*g)))
for name,M,gamma in [('Xe',.131293,.97),('N2O2',.0300061,.86)]:
 for I in [.100,.099,.060]:
  F=gamma*I*math.sqrt(2*M/NA*2000/e);md=3*mol_sccm*M
  res['beam'].append(dict(gas=name,I_A=I,flow_kg_s=md,F_mN=1000*F,Isp_s=F/md/g))
for fluid,M,q in [('Xenon',.131293,4),('N2O2',.0300061,8)]:
 use=mol_sccm*q*500*3600*M
 # residual mass evaluated at 2 bar and 20 C
 rhores=brentq(lambda rho:pressure(fluid,293.15,rho)-2e5,.001,100)
 load=1.1*use+V*rhores;rho=load/V
 d=dict(usable_g=use*1000,residual_g=V*rhores*1000,loaded_g=load*1000,density_kg_m3=rho)
 d['pressure_bar']={str(t):pressure(fluid,t+273.15,rho)/1e5 for t in [-10,20,30,40,50]}
 if fluid=='Xenon':
  d['dew_C']=brentq(lambda T:CP.PropsSI('Dmass','T',T,'Q',1,'Xenon')-rho,200,289.72)-273.15
 else:
  maxrho=brentq(lambda rho:pressure(fluid,323.15,rho)-300e5,100,500)
  d['load_at_300bar_50C_g']=maxrho*V*1000
  d['nominal_hours_at_limited_load']=(maxrho*V-V*rhores)/(mol_sccm*q*M)/3600
  d['hours_with_10pct_margin']=d['nominal_hours_at_limited_load']/1.1
 res['storage'][fluid]=d
# Discrete 380-turn coil, 90 mm clear bore at first wire inner surface; no former allowance.
dw=.001032; rr=.045+(np.arange(10)+.5)*dw; zz=(np.arange(38)-18.5)*dw
r,z=np.meshgrid(rr,zz); mu=4*np.pi*1e-7
BperA=lambda x: np.sum(mu*r*r/(2*(r*r+(x-z)**2)**1.5))
I=.005/BperA(0);length=float(np.sum(2*np.pi*r));area=np.pi*.000912**2/4
Rc=1.724e-8*length/area;P=I*I*Rc
thermal={str(eps):brentq(lambda T:I*I*Rc*(1+.00393*(T-293.15))-eps*5.670374419e-8*.0204*(T**4-293.15**4),294,600)-273.15 for eps in [.3,.8]}
res['coil']=dict(turns=380,current_A=I,resistance_ohm=Rc,power_W=P,copper_kg=length*area*8960,length_m=length,edge_mT=BperA(.02)*I*1000,thermal_C=thermal)
res['checks']=dict(compression_K=293.15*274**(2/7),yield_interpolation=.95*2.3**(math.log(2.4/.95)/math.log(5)))
(out/'results.json').write_text(json.dumps(res,indent=2,allow_nan=False))
fig,axs=plt.subplots(1,2,figsize=(10,4.1))
x=np.linspace(-.04,.04,301)
axs[0].plot(x*1000,[BperA(t)*I*1000 for t in x],label='1.125 A')
axs[0].plot(x*1000,[BperA(t)*I*.1*1000 for t in x],label='0.1125 A')
axs[0].axvspan(-20,20,alpha=.08);axs[0].set(xlabel='Axial position (mm)',ylabel='Axial field (mT)');axs[0].legend(loc='upper right',fontsize=8)
ins=axs[0].inset_axes([.25,.25,.5,.25]);xx=np.linspace(-.02,.02,101)
for cur,label in [(I*.1,'0.1125 A'),(.0005/BperA(.02),'0.1341 A')]: ins.plot(xx*1000,[BperA(t)*cur*1000 for t in xx],label=label)
ins.axhline(.5,color='gray',ls=':');ins.set_ylim(.4,.65);ins.tick_params(labelsize=6);ins.legend(fontsize=6,loc='upper center')
for fluid,label in [('Xenon','Xenon'),('N2O2','50:50 N₂/O₂')]:
 ts=np.sort(np.r_[np.linspace(-10,50,201),res['storage']['Xenon']['dew_C']]);axs[1].plot(ts,[pressure(fluid,t+273.15,res['storage'][fluid]['density_kg_m3'])/1e5 for t in ts],label=label)
dew=res['storage']['Xenon']['dew_C'];axs[1].axvline(dew,color='gray',ls=':',lw=1);axs[1].text(dew+1,120,'Xe phase boundary\n13.37 °C',fontsize=8)
axs[1].axhline(300,color='gray',ls='--');axs[1].set(xlabel='Temperature (°C)',ylabel='Pressure (bar)');axs[1].legend(loc='upper left',fontsize=8)
fig.tight_layout();fig.savefig(out/'calculations.pdf');print(json.dumps(res,indent=2,allow_nan=False))

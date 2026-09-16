"""CPU checks of the analytical oracle, not GPU qualification."""
import math
import numpy as np
from analyze import reference_cdf, reference_density, moments, analyze

def test_reference():
    edges = np.geomspace(math.exp(-3), math.exp(3), 20001)
    for t in (0.0, 100.0, 1000.0):
        mu, var = moments(t)
        assert abs(float(reference_cdf(np.array([math.exp(mu)]), t)[0]) - 0.5) < 1e-15
        mass = np.diff(reference_cdf(edges, t))
        u = np.log(np.sqrt(edges[:-1]*edges[1:]))
        assert abs(mass.sum()-1) < 1e-8
        assert abs(np.sum(mass*u)-mu) < 1e-7
        assert abs(np.sum(mass*(u-mu)**2)-var) < 1e-7
        r = np.sqrt(edges[:-1]*edges[1:])
        integrated = reference_density(r,t)*np.pi*np.diff(edges**2)
        assert np.max(np.abs(integrated-mass)) < 1e-8
    # Directly check the cylindrical variable-D PDE at representative points.
    r=np.array([0.6,0.9,1.2,1.7]); t=300.; dr=1e-4; dt=1e-2
    f=lambda x:reference_density(x,t)
    temporal=(reference_density(r,t+dt)-reference_density(r,t-dt))/(2*dt)
    flux=lambda x: x*(1e-4*x*x)*(f(x+dr)-f(x-dr))/(2*dr)
    spatial=(flux(r+dr)-flux(r-dr))/(2*dr*r)
    assert np.max(np.abs(temporal-spatial)) < 1e-8

def test_analysis_io():
    import tempfile
    import json
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        output=Path(tmp)
        (output/'variables.txt').write_text('[PARAMETERS]\nn_p=1048576\nidx_q=0.5\nalpha=0.04\naspr_0=0.05\nschmidt_r=1\ndt_out=100\ndt_max=1\nsave_max=10\ny_min=0.049787068\ny_max=20.085537\n')
        n=1048576; state=np.zeros((n,6)); state[:,2]=math.pi/2
        rng=np.random.default_rng(8)
        for frame in range(11):
            mu,var=moments(frame*100)
            state[:,1]=np.exp(mu+math.sqrt(var)*rng.standard_normal(n))
            state.tofile(output/f'particle_{frame:05d}.dat')
        result=json.loads(analyze(output).read_text())
        assert len(result['records'])==11
        assert all(x['cdf_sup_error']<0.006 for x in result['records'])
        assert all(x['max_abs_velocity']==0 for x in result['records'])

if __name__ == '__main__':
    test_reference()
    test_analysis_io()
    print('Analytical-reference checks passed; no GPU execution.')

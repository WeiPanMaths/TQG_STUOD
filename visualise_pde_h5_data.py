from firedrake import *
import sys
from firedrake_utility import TorusMeshHierarchy
from utility import Workspace
#from stqg.generate_initial_ensemble import STQGEnsembleGenerator
from stqg.solver import STQGSolver 
from tqg.example2 import TQGExampleTwo as Example2params

desired_spatial_rank = 1
ensemble = ensemble.Ensemble(COMM_WORLD, desired_spatial_rank)
spatial_comm = ensemble.comm
ensemble_comm = ensemble.ensemble_comm

prefix = 'c'
def pde_data_fname(wspace, file_index, sub_dir=''):
    return wspace.output_name("{}pde_data_{}".format(prefix, file_index), sub_dir) 

if __name__ == "__main__":
    nx = 512

    if prefix=='c':
        nx = 32
    else:
        nx = 512

    print(nx)
    #nx = 512 
    #cnx = 32 #64
    T = 0.1
    dt = 0.0001
    dump_freq =10 
    solve_flag = False
    write_visual = False 
    batch_id = 0

    #print("batch_id: ", batch_id, ", ensemble_member_id: ",  ensemble_comm.rank + batch_id*ensemble_comm.size)

    ensemble_size = ensemble_comm.size 

    mesh = TorusMeshHierarchy(nx, nx, 1., 1., 0, period="y",comm=spatial_comm).get_fine_mesh()
    
    # It doesn't matter what the actual params are, we just need a param object to pass to the Solver constructor
    angry_dolphin_params = Example2params(T, dt, mesh, bc='x', alpha=None)
    
    ensemble_generator = STQGSolver(angry_dolphin_params)

    file_work_space = Workspace("/home/wpan1/Data2/PythonProjects/DataAssimilationAngryDolphin")

    output_file = File(file_work_space.output_name("{}pde_data.pvd".format(prefix), "{}PDESolution/visuals".format(prefix)), comm=spatial_comm) 
    
    dt = 0.0001
    for findex in range(0,500):
        print(findex)
        ensemble_generator.load_initial_conditions_from_file(pde_data_fname(file_work_space, findex, "{}PDESolution".format(prefix)), spatial_comm)
        output_file.write(ensemble_generator.initial_cond, ensemble_generator.initial_b, ensemble_generator.psi0, ensemble_generator.ssh,  time=round(dt*dump_freq*findex,5))


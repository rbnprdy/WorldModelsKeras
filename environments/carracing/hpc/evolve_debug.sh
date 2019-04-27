# Your job will use 1 node, 28 cores, and 168gb of memory total.
#PBS -q standard
#PBS -l select=1:ncpus=3:mem=168gb:pcmem=6gb

### Specify a name for the job
#PBS -N debug

### Specify the group name
#PBS -W group_list=akoglu

### CPUtime required in hhh:mm:ss.
### Leading 0's can be omitted e.g 48:0:0 sets 48 hours
#PBS -l cput=3:0:0

### Walltime is how long your job will run
#PBS -l walltime=1:0:0

### Email me at beginning, end, and abnormal end
#PBS -m bea
#PBS -M rubenpurdy@email.arizona.edu

### EXPERIMENTS

module load openmpi
module load singularity

cd /extra/rubenpurdy/WorldModelsKeras/environments/carracing

date
mpirun -np 3 singularity exec /extra/rubenpurdy/images/gym_cpu.simg xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python evolve_controller.py carracing -n 2 -t 1 -e 1
date
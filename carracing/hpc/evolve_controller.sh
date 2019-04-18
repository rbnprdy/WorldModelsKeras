# Your job will use 1 node, 28 cores, and 168gb of memory total.
#PBS -q standard
#PBS -l select=1:ncpus=28:mem=168gb:pcmem=6gb:ngpus=1

### Specify a name for the job
#PBS -N wm_evolve

### Specify the group name
#PBS -W group_list=akoglu

### Used if job requires partial node only
#PBS -l place=pack:exclhost

### CPUtime required in hhh:mm:ss.
### Leading 0's can be omitted e.g 48:0:0 sets 48 hours
#PBS -l cput=280:00:00

### Walltime is how long your job will run
#PBS -l walltime=10:00:00

### Email me at beginning, end, and abnormal end
#PBS -m bea
#PBS -M rubenpurdy@email.arizona.edu

### EXPERIMENTS

module load singularity

cd /extra/rubenpurdy/WorldModelsKeras/carracing

date
singularity exec --nv /extra/rubenpurdy/images/gym.simg xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python evolve_controller.py --num_worker 28 --num_worker_trial 2 --num_episode 4
date
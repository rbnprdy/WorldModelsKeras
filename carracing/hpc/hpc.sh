# Your job will use 1 node, 28 cores, and 168gb of memory total.
#PBS -q standard
#PBS -l select=1:ncpus=28:mem=168gb:pcmem=6gb:ngpus=1

### Specify a name for the job
#PBS -N world_models

### Specify the group name
#PBS -W group_list=akoglu

### Used if job requires partial node only
#PBS -l place=pack:exclhost

### CPUtime required in hhh:mm:ss.
### Leading 0's can be omitted e.g 48:0:0 sets 48 hours
#PBS -l cput=23:20:00

### Walltime is how long your job will run
#PBS -l walltime=00:50:00

### EXPERIMENTS

module load singularity

cd /extra/rubenpurdy/WorldModelsKeras/carracing

date
singularity exec /extra/rubenpurdy/images/gym.simg python extract.py data/ --num_trials 4 --num_frames 100
date
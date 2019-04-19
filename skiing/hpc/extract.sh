# Your job will use 1 node, 28 cores, and 168gb of memory total.
#PBS -q standard
#PBS -l select=1:ncpus=28:mem=168gb:pcmem=6gb:ngpus=1

### Specify a name for the job
#PBS -N skiing_extract

### Specify the group name
#PBS -W group_list=akoglu

### Used if job requires partial node only
#PBS -l place=pack:exclhost

### CPUtime required in hhh:mm:ss.
### Leading 0's can be omitted e.g 48:0:0 sets 48 hours
#PBS -l cput=70:00:00

### Walltime is how long your job will run
#PBS -l walltime=2:30:00

### Email me at beginning, end, and abnormal end
#PBS -m bea
#PBS -M rubenpurdy@email.arizona.edu

### EXPERIMENTS

module load singularity

cd /extra/rubenpurdy/WorldModelsKeras/skiing

date
./extract.bash
date
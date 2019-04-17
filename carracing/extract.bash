for i in `seq 1 28`;
do
  echo worker $i
  # 358 trials per core with 28 cores will generate 10,024 trials total  
  xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python extract.py data/ --num_trials 358 &
  sleep 1.0
done
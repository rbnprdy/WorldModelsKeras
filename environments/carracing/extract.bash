for i in `seq 1 28`;
do
  echo worker $i
  # 393 trials per core with 28 cores will generate 11,004 trials total (10,000 training, 1,000 validation)
  singularity exec /extra/rubenpurdy/images/gym.simg xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python extract.py data/ --num_trials 393 &
  sleep 1.0
done
wait
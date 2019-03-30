for i in `seq 1 64`;
do
  echo worker $i
  # on cloud:
  #DISPLAY=:0 python extract.py record &
  python extract.py -o data/$i.h5 &
  sleep 1.0
done
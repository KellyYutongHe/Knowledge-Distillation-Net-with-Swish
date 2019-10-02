echo "START 0110"
python kd.py --alpha 0.1 --temperature 10 | tee log/kd_01_10.log
echo "START 0510"
python kd.py --alpha 0.5 --temperature 10 | tee log/kd_05_10.log
echo "START 0910"
python kd.py --alpha 0.9 --temperature 10 | tee log/kd_09_10.log
echo "START 015"
python kd.py --alpha 0.1 --temperature 5 | tee log/kd_01_5.log
echo "START 055"
python kd.py --alpha 0.5 --temperature 5 | tee log/kd_05_5.log
echo "START 095"
python kd.py --alpha 0.9 --temperature 5 | tee log/kd_09_5.log
echo "DONE"

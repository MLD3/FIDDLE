# python prepare_data_mortality.py 2>&1 | tee log/prepare_data_mortality.log
# python prepare_data.py --outcome='ARF' --T='4.0' | tee log/prepare_data_ARF_4h.log
# python prepare_data.py --outcome='Shock' --T='4.0' | tee log/prepare_data_Shock_4h.log
python prepare_data.py --outcome='ARF' --T='12.0' | tee log/prepare_data_ARF_12h.log
python prepare_data.py --outcome='Shock' --T='12.0' | tee log/prepare_data_Shock_12h.log

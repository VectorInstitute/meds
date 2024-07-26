"""Tests the add time derived measurements script.

Set the bash env variable `DO_USE_LOCAL_SCRIPTS=1` to use the local py files, rather than the installed
scripts.
"""


from .transform_tester_base import (
    ADD_TIME_DERIVED_MEASUREMENTS_SCRIPT,
    single_stage_transform_tester,
)
from .utils import parse_meds_csvs

AGE_CALCULATION_STR = """
See `add_time_derived_measurements.py` for the source of the constant value.

```python
from datetime import datetime
seconds_in_year = 31556926.080000002
timestamps_by_dob = {
    "12/28/1980, 00:00:00": [
        "05/11/2010, 17:41:51",
        "05/11/2010, 17:48:48",
        "05/11/2010, 18:25:35",
        "05/11/2010, 18:57:18",
        "05/11/2010, 19:27:19",
    ],
    "06/20/1978, 00:00:00": [
        "06/20/2010, 19:23:52",
        "06/20/2010, 19:25:32",
        "06/20/2010, 19:45:19",
        "06/20/2010, 20:12:31",
        "06/20/2010, 20:24:44",
        "06/20/2010, 20:41:33",
        "06/20/2010, 20:50:04",
    ],
    "03/09/1978, 00:00:00": [
        "05/26/2010, 02:30:56",
        "05/26/2010, 04:51:52",
    ],
    "03/28/1976, 00:00:00": [
        "02/05/2010, 05:55:39",
        "02/05/2010, 07:02:30",
    ],
    "12/19/1988, 00:00:00": [
        "01/03/2010, 06:27:59",
        "01/03/2010, 08:22:13",
    ],
    "07/20/1986, 00:00:00": [
        "06/03/2010, 14:54:38",
        "06/03/2010, 15:39:49",
        "06/03/2010, 16:20:49",
        "06/03/2010, 16:44:26",
    ],
}

ts_fmt = "%m/%d/%Y, %H:%M:%S"

ages_by_dob = {}
for k, ts in timestamps_by_dob.items():
    print(k)
    dob_ts = datetime.strptime(k, ts_fmt)
    for event_ts_str in ts:
        event_ts = datetime.strptime(event_ts_str, ts_fmt)
        age = (event_ts - dob_ts).total_seconds() / seconds_in_year
        print(f'  - {age}')
```

This returns
```
12/28/1980, 00:00:00
  - 29.36883360091833
  - 29.36884681513314
  - 29.36891675223647
  - 29.36897705595538
  - 29.369034127420306
06/20/1978, 00:00:00
  - 32.002896271955265
  - 32.00289944083172
  - 32.00293705539522
  - 32.002988771458945
  - 32.00301199932335
  - 32.003043973286765
  - 32.00306016624544
03/09/1978, 00:00:00
  - 32.214527277556684
  - 32.21479523774959
03/28/1976, 00:00:00
  - 33.86039997340577
  - 33.860527077040324
12/19/1988, 00:00:00
  - 21.04157031380922
  - 21.04178750860134
07/20/1986, 00:00:00
  - 23.873531791091356
  - 23.873617699332012
  - 23.873695653692767
  - 23.873740556672114
```
"""

WANT_TRAIN_0 = """
patient_id,timestamp,code,numerical_value
239684,,EYE_COLOR//BROWN,
239684,,HEIGHT,175.271115221764
239684,"12/28/1980, 00:00:00","TIME_OF_DAY//[00,06)",
239684,"12/28/1980, 00:00:00",DOB,
239684,"05/11/2010, 17:41:51","TIME_OF_DAY//[12,18)",
239684,"05/11/2010, 17:41:51",AGE,29.36883360091833
239684,"05/11/2010, 17:41:51",ADMISSION//CARDIAC,
239684,"05/11/2010, 17:41:51",HR,102.6
239684,"05/11/2010, 17:41:51",TEMP,96.0
239684,"05/11/2010, 17:48:48","TIME_OF_DAY//[12,18)",
239684,"05/11/2010, 17:48:48",AGE,29.36884681513314
239684,"05/11/2010, 17:48:48",HR,105.1
239684,"05/11/2010, 17:48:48",TEMP,96.2
239684,"05/11/2010, 18:25:35","TIME_OF_DAY//[18,24)",
239684,"05/11/2010, 18:25:35",AGE,29.36891675223647
239684,"05/11/2010, 18:25:35",HR,113.4
239684,"05/11/2010, 18:25:35",TEMP,95.8
239684,"05/11/2010, 18:57:18","TIME_OF_DAY//[18,24)",
239684,"05/11/2010, 18:57:18",AGE,29.36897705595538
239684,"05/11/2010, 18:57:18",HR,112.6
239684,"05/11/2010, 18:57:18",TEMP,95.5
239684,"05/11/2010, 19:27:19","TIME_OF_DAY//[18,24)",
239684,"05/11/2010, 19:27:19",AGE,29.369034127420306
239684,"05/11/2010, 19:27:19",DISCHARGE,
1195293,,EYE_COLOR//BLUE,
1195293,,HEIGHT,164.6868838269085
1195293,"06/20/1978, 00:00:00","TIME_OF_DAY//[00,06)",
1195293,"06/20/1978, 00:00:00",DOB,
1195293,"06/20/2010, 19:23:52","TIME_OF_DAY//[18,24)",
1195293,"06/20/2010, 19:23:52",AGE,32.002896271955265
1195293,"06/20/2010, 19:23:52",ADMISSION//CARDIAC,
1195293,"06/20/2010, 19:23:52",HR,109.0
1195293,"06/20/2010, 19:23:52",TEMP,100.0
1195293,"06/20/2010, 19:25:32","TIME_OF_DAY//[18,24)",
1195293,"06/20/2010, 19:25:32",AGE,32.00289944083172
1195293,"06/20/2010, 19:25:32",HR,114.1
1195293,"06/20/2010, 19:25:32",TEMP,100.0
1195293,"06/20/2010, 19:45:19","TIME_OF_DAY//[18,24)",
1195293,"06/20/2010, 19:45:19",AGE,32.00293705539522
1195293,"06/20/2010, 19:45:19",HR,119.8
1195293,"06/20/2010, 19:45:19",TEMP,99.9
1195293,"06/20/2010, 20:12:31","TIME_OF_DAY//[18,24)",
1195293,"06/20/2010, 20:12:31",AGE,32.002988771458945
1195293,"06/20/2010, 20:12:31",HR,112.5
1195293,"06/20/2010, 20:12:31",TEMP,99.8
1195293,"06/20/2010, 20:24:44","TIME_OF_DAY//[18,24)",
1195293,"06/20/2010, 20:24:44",AGE,32.00301199932335
1195293,"06/20/2010, 20:24:44",HR,107.7
1195293,"06/20/2010, 20:24:44",TEMP,100.0
1195293,"06/20/2010, 20:41:33","TIME_OF_DAY//[18,24)",
1195293,"06/20/2010, 20:41:33",AGE,32.003043973286765
1195293,"06/20/2010, 20:41:33",HR,107.5
1195293,"06/20/2010, 20:41:33",TEMP,100.4
1195293,"06/20/2010, 20:50:04","TIME_OF_DAY//[18,24)",
1195293,"06/20/2010, 20:50:04",AGE,32.00306016624544
1195293,"06/20/2010, 20:50:04",DISCHARGE,
"""

# All patients in this shard had only 4 events.
WANT_TRAIN_1 = """
patient_id,timestamp,code,numerical_value
68729,,EYE_COLOR//HAZEL,
68729,,HEIGHT,160.3953106166676
68729,"03/09/1978, 00:00:00","TIME_OF_DAY//[00,06)",
68729,"03/09/1978, 00:00:00",DOB,
68729,"05/26/2010, 02:30:56","TIME_OF_DAY//[00,06)",
68729,"05/26/2010, 02:30:56",AGE,32.214527277556684
68729,"05/26/2010, 02:30:56",ADMISSION//PULMONARY,
68729,"05/26/2010, 02:30:56",HR,86.0
68729,"05/26/2010, 02:30:56",TEMP,97.8
68729,"05/26/2010, 04:51:52","TIME_OF_DAY//[00,06)",
68729,"05/26/2010, 04:51:52",AGE,32.21479523774959
68729,"05/26/2010, 04:51:52",DISCHARGE,
814703,,EYE_COLOR//HAZEL,
814703,,HEIGHT,156.48559093209357
814703,"03/28/1976, 00:00:00","TIME_OF_DAY//[00,06)",
814703,"03/28/1976, 00:00:00",DOB,
814703,"02/05/2010, 05:55:39","TIME_OF_DAY//[00,06)",
814703,"02/05/2010, 05:55:39",AGE,33.86039997340577
814703,"02/05/2010, 05:55:39",ADMISSION//ORTHOPEDIC,
814703,"02/05/2010, 05:55:39",HR,170.2
814703,"02/05/2010, 05:55:39",TEMP,100.1
814703,"02/05/2010, 07:02:30","TIME_OF_DAY//[06,12)",
814703,"02/05/2010, 07:02:30",AGE,33.860527077040324
814703,"02/05/2010, 07:02:30",DISCHARGE,
"""

WANT_TUNING_0 = """
patient_id,timestamp,code,numerical_value
754281,,EYE_COLOR//BROWN,
754281,,HEIGHT,166.22261567137025
754281,"12/19/1988, 00:00:00","TIME_OF_DAY//[00,06)",
754281,"12/19/1988, 00:00:00",DOB,
754281,"01/03/2010, 06:27:59","TIME_OF_DAY//[06,12)",
754281,"01/03/2010, 06:27:59",AGE,21.04157031380922
754281,"01/03/2010, 06:27:59",ADMISSION//PULMONARY,
754281,"01/03/2010, 06:27:59",HR,142.0
754281,"01/03/2010, 06:27:59",TEMP,99.8
754281,"01/03/2010, 08:22:13","TIME_OF_DAY//[06,12)",
754281,"01/03/2010, 08:22:13",AGE,21.04178750860134
754281,"01/03/2010, 08:22:13",DISCHARGE,
"""

WANT_HELD_OUT_0 = """
patient_id,timestamp,code,numerical_value
1500733,,EYE_COLOR//BROWN,
1500733,,HEIGHT,158.60131573580904
1500733,"07/20/1986, 00:00:00","TIME_OF_DAY//[00,06)",
1500733,"07/20/1986, 00:00:00",DOB,
1500733,"06/03/2010, 14:54:38","TIME_OF_DAY//[12,18)",
1500733,"06/03/2010, 14:54:38",AGE,23.873531791091356
1500733,"06/03/2010, 14:54:38",ADMISSION//ORTHOPEDIC,
1500733,"06/03/2010, 14:54:38",HR,91.4
1500733,"06/03/2010, 14:54:38",TEMP,100.0
1500733,"06/03/2010, 15:39:49","TIME_OF_DAY//[12,18)",
1500733,"06/03/2010, 15:39:49",AGE,23.873617699332012
1500733,"06/03/2010, 15:39:49",HR,84.4
1500733,"06/03/2010, 15:39:49",TEMP,100.3
1500733,"06/03/2010, 16:20:49","TIME_OF_DAY//[12,18)",
1500733,"06/03/2010, 16:20:49",AGE,23.873695653692767
1500733,"06/03/2010, 16:20:49",HR,90.1
1500733,"06/03/2010, 16:20:49",TEMP,100.1
1500733,"06/03/2010, 16:44:26","TIME_OF_DAY//[12,18)",
1500733,"06/03/2010, 16:44:26",AGE,23.873740556672114
1500733,"06/03/2010, 16:44:26",DISCHARGE,
"""

WANT_SHARDS = parse_meds_csvs(
    {
        "train/0": WANT_TRAIN_0,
        "train/1": WANT_TRAIN_1,
        "tuning/0": WANT_TUNING_0,
        "held_out/0": WANT_HELD_OUT_0,
    }
)


def test_add_time_derived_measurements():
    single_stage_transform_tester(
        transform_script=ADD_TIME_DERIVED_MEASUREMENTS_SCRIPT,
        stage_name="add_time_derived_measurements",
        transform_stage_kwargs={"age": {"DOB_code": "DOB"}},
        want_outputs=WANT_SHARDS,
    )
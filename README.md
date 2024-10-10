# Problematic-Internet-Use

`
Based on physical activity and fitness data identify early signs of problematic internet use.
`

- The Healthy Brain Network (HBN) dataset is a clinical sample of about five-thousand 5-22 year-olds who have undergone both clinical and research screenings.
- The objective of the HBN study is to find biological markers that will improve the diagnosis and treatment of mental health and learning disorders from an objective biological perspective.
- wo elements of this study are being used for this competition: physical activity data (wrist-worn accelerometer data, fitness assessments and questionnaires) and internet usage behavior data.
- The goal of this competition is to predict from this data a participant's Severity Impairment Index (sii), a standard measure of problematic internet use.
- The full test set comprises about 3800 instances.

- The competition data is compiled into two sources, parquet files containing the accelerometer (actigraphy) series and csv files containing the remaining tabular data.
- The majority of measures are missing for most participants.
- In particular, the target sii is missing for a portion of the participants in the training set.
- You may wish to apply non-supervised learning techniques to this data. The sii value is present for all instances in the test set.


- The majority of measures are missing for most participants. In particular, the target sii is missing for a portion of the participants in the training set.
- You may wish to apply non-supervised learning techniques to this data. The sii value is present for all instances in the test set.

**HBN Instruments**

- The tabular data in `train.csv` and `test.csv` comprises measurements from a variety of instruments.
- The fields within each instrument are described in `data_dictionary.csv`. These instruments are:

**Demographics** - Information about age and sex of participants.
**Internet Use** - Number of hours of using computer/internet per day.
**Children's Global Assessment Scale** - Numeric scale used by mental health clinicians to rate the general functioning of youths under the age of 18.
**Physical Measures** - Collection of blood pressure, heart rate, height, weight and waist, and hip measurements.
**FitnessGram Vitals and Treadmill** - Measurements of cardiovascular fitness assessed using the NHANES treadmill protocol.
**FitnessGram Child** - Health related physical fitness assessment measuring five different parameters including aerobic capacity, muscular strength, muscular endurance, flexibility, and body composition.
**Bio-electric Impedance Analysis** - Measure of key body composition elements, including BMI, fat, muscle, and water content.
**Physical Activity Questionnaire** - Information about children's participation in vigorous activities over the last 7 days.
**Sleep Disturbance Scale** - Scale to categorize sleep disorders in children.
**Actigraphy** - Objective measure of ecological physical activity through a research-grade biotracker.
**Parent-Child Internet Addiction Test** - 20-item scale that measures characteristics and behaviors associated with compulsive use of the Internet including compulsivity, escapism, and dependency.

Note in particular the field **PCIAT-PCIAT_Total**. The target sii for this competition is derived from this field as described in the data dictionary: 0 for None, 1 for Mild, 2 for Moderate, and 3 for Severe. Additionally, each participant has been assigned a unique identifier id.

```
>>> df2.columns
Index(['id', 'Basic_Demos-Enroll_Season', 'Basic_Demos-Age', 'Basic_Demos-Sex',
       'CGAS-Season', 'CGAS-CGAS_Score', 'Physical-Season', 'Physical-BMI',
       'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference',
       'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP',
       'Fitness_Endurance-Season', 'Fitness_Endurance-Max_Stage',
       'Fitness_Endurance-Time_Mins', 'Fitness_Endurance-Time_Sec',
       'FGC-Season', 'FGC-FGC_CU', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND',
       'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD', 'FGC-FGC_GSD_Zone', 'FGC-FGC_PU',
       'FGC-FGC_PU_Zone', 'FGC-FGC_SRL', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR',
       'FGC-FGC_SRR_Zone', 'FGC-FGC_TL', 'FGC-FGC_TL_Zone', 'BIA-Season',
       'BIA-BIA_Activity_Level_num', 'BIA-BIA_BMC', 'BIA-BIA_BMI',
       'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 'BIA-BIA_FFM',
       'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat', 'BIA-BIA_Frame_num',
       'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM',
       'BIA-BIA_TBW', 'PAQ_A-Season', 'PAQ_A-PAQ_A_Total', 'PAQ_C-Season',
       'PAQ_C-PAQ_C_Total', 'PCIAT-Season', 'PCIAT-PCIAT_01', 'PCIAT-PCIAT_02',
       'PCIAT-PCIAT_03', 'PCIAT-PCIAT_04', 'PCIAT-PCIAT_05', 'PCIAT-PCIAT_06',
       'PCIAT-PCIAT_07', 'PCIAT-PCIAT_08', 'PCIAT-PCIAT_09', 'PCIAT-PCIAT_10',
       'PCIAT-PCIAT_11', 'PCIAT-PCIAT_12', 'PCIAT-PCIAT_13', 'PCIAT-PCIAT_14',
       'PCIAT-PCIAT_15', 'PCIAT-PCIAT_16', 'PCIAT-PCIAT_17', 'PCIAT-PCIAT_18',
       'PCIAT-PCIAT_19', 'PCIAT-PCIAT_20', 'PCIAT-PCIAT_Total', 'SDS-Season',
       'SDS-SDS_Total_Raw', 'SDS-SDS_Total_T', 'PreInt_EduHx-Season',
       'PreInt_EduHx-computerinternet_hoursday', 'sii'],
      dtype='object')
```


**Actigraphy Files and Field Descriptions**

During their participation in the HBN study, some participants were given an accelerometer to wear for up to 30 days continually while at home and going about their regular daily lives.

series_{train|test}.parquet/id={id} - Series to be used as training data, partitioned by id. Each series is a continuous recording of accelerometer data for a single subject spanning many days.

**id** - The patient identifier corresponding to the id field in train/test.csv.
**step** - An integer timestep for each observation within a series.
**X, Y, Z** - Measure of acceleration, in g, experienced by the wrist-worn watch along each standard axis.
**enmo** - As calculated and described by the wristpy package, ENMO is the Euclidean Norm Minus One of all accelerometer signals (along each of the x-, y-, and z-axis, measured in g-force) with negative values rounded to zero. Zero values are indicative of periods of no motion. While no standard measure of acceleration exists in this space, this is one of the several commonly computed features.
**anglez** - As calculated and described by the wristpy package, Angle-Z is a metric derived from individual accelerometer components and refers to the angle of the arm relative to the horizontal plane.
**non-wear_flag** - A flag (0: watch is being worn, 1: the watch is not worn) to help determine periods when the watch has been removed, based on the GGIR definition, which uses the standard deviation and range of the accelerometer data.
**light** - Measure of ambient light in lux. See ​​here for details.
**battery_voltage** - A measure of the battery voltage in mV.
**time_of_day** - Time of day representing the start of a 5s window that the data has been sampled over, with format %H:%M:%S.%9f.
**weekday** - The day of the week, coded as an integer with 1 being Monday and 7 being Sunday.
**quarter** - The quarter of the year, an integer from 1 to 4.
**relative_date_PCIAT** - The number of days (integer) since the PCIAT test was administered (negative days indicate that the actigraphy data has been collected before the test was administered).


**sample_submission.csv** - A sample submission file in the correct format. See the Evaluation page for more details.

```csv
id,sii
000046df,0
000089ff,1
00012558,2
00017ccd,3
```

## Hand-by-hand guidance for prepare dataset

### ScanNet V2
- Download the ScanNet V2 dataset and go to its dir, which has the following files
```bash
scannetv2-labels.combined.tsv # label for train and val set
scans                         # scans for train and val set
scans_test                    # scans for test set
tasks
```
- Preprocess the train and val files as follows
```bash
python -u scannet/preprocess_scannet.py \
  --input PATH_TO_YOUR_SCANNETV2/scans \
  --output SOME_PATH/scannet_fully_supervised_preprocessed \
  --splits train # train includes both train & val
```
- Copy the split files to your preprocessed dir
```bash
cp -r scannet/splits SOME_PATH/scannet_fully_supervised_preprocessed/
```
- Generate the label ids, which are used to provide the ``weak (sparse) labels``
- Two types of sparse labels:
  - Absolute points: 20,50,100,200 points from [Official ScanNet Data Efficient Benchmark](https://kaldir.vc.in.tum.de/scannet_benchmark/data_efficient/documentation)
  - Percentage points: 0.01%, 0.1% generated by our script (:fire: the script will be released later)
```
cp -r scannet/points SOME_PATH/scannet_fully_supervised_preprocessed/
```

<br>

### Stanford (S3DIS)
- Download S3DIS data by filling this [Google form](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1). Download the Stanford3dDataset_v1.2.zip file and unzip it.
- Run preprocessing code for S3DIS as follows:
```bash
python prepare_dataset/stanford/preproceess_stanford.py --input ${PATH_TO_S3DIS_DIR} --output_root ${PROCESSED_S3DIS_DIR}
```
- Copy the split files to your preprocessed dir
```bash
cp -r prepare_dataset/stanford/splits SOME_PATH/stanford_fully_supervised_preprocessed/
```
- Generate the label ids, which are used to provide the ``weak (sparse) labels``
- Only one type of sparse labels:
  - Percentage points: 0.01%, 0.02%, 0.1%, 0.2%, 10%, generated by our script (:fire: the script will be released later)
```bash
cp -r prepare_dataset/stanford/points SOME_PATH/stanford_fully_supervised_preprocessed/
```

<br>

### SemanticKITTI-FoV (Front view that includes both RGB and XYZ)
:fire: Coming soon!

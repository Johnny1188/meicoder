## Data
`<DATA_PATH>/mouse_v1_sensorium22` should contain all the `.zip` files downloaded from the [SENSORIUM 2022 data repository](https://gin.g-node.org/cajal/Sensorium2022):
  - `static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip` (data key `26872-17-20`; used only for synthetic data generation and encoder training)
  - `static27204-5-13-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip` (data key `27204-5-13`; used only for synthetic data generation and encoder training)
  - `static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip` (dataset **M-1**; data key `21067-10-18`)
  - `static22846-10-16-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip` (data key `22846-10-16`; mouse ID 2)
  - `static23343-5-17-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip` (data key `23343-5-17`; mouse ID 3)
  - `static23656-14-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip` (data key `23656-14-22`; mouse ID 4)
  - `static23964-4-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip` (data key `23964-4-22`; mouse ID 5)

While the data with the data key `21067-10-18` correspond to the dataset **M-1**, data from the last 5 datasets above represent the combined dataset **M-All**.

When first running the code, the `.zip` files will be automatically extracted to the same directory.

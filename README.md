# 임베디드소프트웨어 프로젝트
 
 ## 여러 Convolution 성능 비교와 Pruning을 이용한 Bp segmentation

**팀 하하크크**

### 레포지토리 구성

📦embed <br>
 ┣ 📂Kaggle <br>
 ┃ ┣ 📜10percent-pruned.ipynb <br>
 ┃ ┣ 📜15percent-pruned.ipynb <br>
 ┃ ┣ 📜20percent-pruned.ipynb <br>
 ┃ ┣ 📜25percent-pruned.ipynb <br>
 ┃ ┣ 📜DataAugment.ipynb <br>
 ┃ ┣ 📜DataAugment_Separable.ipynb <br>
 ┃ ┣ 📜data-augment-dilated-convolution.ipynb <br>
 ┃ ┗ 📜dataaugment-dilatedconvolution-rate_same.ipynb <br>
 ┣ 📂split_datasets <br>
 ┃ ┣ 📂test <br>
 ┃ ┣ 📂train <br>
 ┃ ┗ 📂val <br>
 ┣ 📂weights <br>
 ┃ ┣ 📜10pruned_unet.keras <br>
 ┃ ┣ 📜15pruned_unet.keras <br>
 ┃ ┣ 📜20pruned_unet.keras <br>
 ┃ ┣ 📜25pruned_unet.keras <br>
 ┃ ┗ 📜final_unet.keras <br>
 ┣ 📜README.md <br>
 ┣ 📜build_unet.py <br>
 ┣ 📜data.py <br>
 ┣ 📜main.py <br>
 ┣ 📜output.py <br>
 ┗ 📜parameter.py <br>

### - **Kaggle 폴더** <br>
  Kaggle notebook에서 진행한 프로젝트 전체적인 flow를 볼 수 있습니다. 데이터 증강, U-net 모델 구조, 학습 결과 그래프, 모델 평가 등이 들어있습니다. 또한, Dice 계수, Loss, inference time, FLOPS 등의 성능 평가 지표 결과도 나와 있습니다. <br>
<br>
랜덤 시드를 고정하여 최대한 성능 결과가 일정하게 나오게 조절하였지만 적은 데이터 수로 인해 각 모델의 편차가 큰 편입니다.

### - **split_datasets 폴더** <br>
  초음파 이미지와 mask 된 bp 신경망 이미지가 들어 있습니다. 적정 비율로 test, train, validation 데이터를 나누었습니다.<br>

### - **weights 폴더** <br>
  라즈베리파이에서 사용할 가중치를 모아둔 폴더입니다. 폴더 내에는 기존 모델을  10% ~ 25% pruning 한 가중치 파일들과 저희 프로젝트 최종 모델 가중치가 들어 있습니다. 라즈베리파이에서 이 가중치를 모델에 load하여 사용할 수 있습니다.<br>
  Kaggle notebook 혹은 다른 환경에서도 가중치 파일 사용이  가능합니다.

### - **README.md** <br>
  현재 읽고 계시는 파일입니다. 프로젝트 관련 파일 안내와 보고서에 작성하지 않은 세부적인 내용 및 결과를 작성합니다. <br>
프로젝트 파일을 관리하여 추후 추가적인 연구나 프로젝트 진행을 원활하고 효율적으로 할 수 있습니다.

### - **그 외 py 파일들** <br>
라즈베리파이에서 빌드할 파일들입니다. 라즈베리파이는 ipynb파일이 빌드되지않아 알아보기 쉽게 분할한 것입니다. <br>
라즈베리파이에는 이 파일들과 가중치, test 데이터를 업로드하고 프로젝트 진행하시면 됩니다.


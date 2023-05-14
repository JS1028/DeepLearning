# DeepLearning

## Deep Learning Study
- in OISL, Yonsei University
- 2022 summer


## Virtual Staining
- in OISL, Yonsei University
- Goal: virtually stain the unstained phase image
- Input phase image: reconstructed by FP
- Output Stained image(RGB): reconstructed by FP

![image](https://user-images.githubusercontent.com/109277474/213646998-183afcfb-2b06-4de0-bdfe-2cf07b695894.png)


### PhaseStain
- https://www.nature.com/articles/s41377-019-0129-y
- supervised learning using GAN
- trained with 256x256 ROIs
- Input: Red or NIR phase image of 'stained' stomach cancer tissue (1ch)
- Output: RGB amplitude image of stained stomach cancer tissue (3ch)
 
#### Stitched image of the training input
![image](https://github.com/JS1028/DeepLearning/assets/109277474/623856d7-a46a-4c5d-af4a-4b67ebfeec2e)

#### Stitched image of the training output


#### Stitched image of the training target
![image](https://github.com/JS1028/DeepLearning/assets/109277474/3245e061-9ac2-48c9-9fed-344e9ff4e1b9)



### NIRStain
- supervised learning using GAN
- trained with 256x256 ROIs
- Input: NIR phase image of 'stained' stomach cancer tissue (1ch)
- Output: RGB amplitude image of stained stomach cancer tissue (3ch)
- test: NIR phase image of 'unstained' stomach cancer tissue


#### Network Structure
![image](https://github.com/JS1028/DeepLearning/assets/109277474/63743cb4-bd80-4571-95c2-48391efa36a8)

#### Stitching result of test output
  
#### Target image used for training
![image](https://github.com/JS1028/DeepLearning/assets/109277474/3245e061-9ac2-48c9-9fed-344e9ff4e1b9)



### CycleStain (ongoing)
- unsupervised learning using CycleGAN
- trained with 256x256 ROIs
- Input: R phase image of 'stained' stomach cancer tissue (1ch)
- Output: RGB amplitude image of 'stained' stomach cancer tissue (3ch)
- Use two pretrained generators to calculate a new loss: Phase2Amp & Amp2Phase

#### Network Structure

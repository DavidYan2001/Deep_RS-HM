## [ICCV 2023] Deep Homography Mixture for Single Image Rolling Shutter Correction

<h4 align="center">Weilong Yan<sup>1</sup>, Robby T. Tan<sup>2</sup>, Bing Zeng<sup>1</sup>, Shuaicheng Liu*<sup>1</sup></center>
<h4 align="center">1. University of Electronic Science and Technology of China</center>
<h4 align="center">2. National University of Singapore</center><br><br>



### Abstract
We present a deep homography mixture motion model for single image rolling shutter correction. Rolling shutter (RS) effects are often caused by row-wise exposure delay in the widely adopted CMOS sensor. Previous methods often require more than one frame for the correction, leading to data quality requirements. Few approaches address the more challenging task of single image RS correction, which often adopt designs like trajectory estimation or long rectangular kernels, to learn the camera motion parameters of an RS image, to restore the global shutter (GS) image. In this work, we adopt a more straightforward method to learn deep homography mixture motion between an RS image and its corresponding GS image, without large solution space or strict restrictions on image features. We show that dividing an image into blocks with a Gaussian weight of block scanlines fits well for the RS setting. Moreover, instead of directly learning the motion mapping, we learn coefficients that assemble several motion bases to produce the correction motion, where these bases are learned from the consecutive frames of natural videos beforehand. Experiments show that our method outperforms existing single RS methods statistically and visually, in both synthesized and real RS images


#### This is the official implementation of our ICCV 2023 paper "[***Deep Homography Mixture for Single Image Rolling Shutter Correction***](https://openaccess.thecvf.com/content/ICCV2023/papers/Yan_Deep_Homography_Mixture_for_Single_Image_Rolling_Shutter_Correction_ICCV_2023_paper.pdf)"

![image](https://github.com/DavidYan2001/Deep_RS-HM/assets/1344482/bc1e2927-b4d5-4fe1-a281-fada51f8a1ca)

![image](https://github.com/DavidYan2001/Deep_RS-HM/assets/1344482/3528f029-e2aa-40eb-95d3-79433ad3be6c)

The synthetic dataset RS-Homo can be generated in https://drive.google.com/drive/folders/1-03hg5S84E0GYJOBPB5USsqYj-Z0cXzY.

### Citations

```
@InProceedings{Yan_2023_ICCV,
    author    = {Yan, Weilong and Tan, Robby T. and Zeng, Bing and Liu, Shuaicheng},
    title     = {Deep Homography Mixture for Single Image Rolling Shutter Correction},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {9868-9877}
}
```
### Contact

Email: Weilong Yan [1092443660ywl@gmail.com](1092443660ywl@gmail.com)


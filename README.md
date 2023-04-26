
1, install python dependencies using requirements.txt file

 "pip install -r requirements.txt"

2, copy the image to test into the project root folder.

3. run the below command

"python server.py --image <name of image>"

ex: python server.py test.png

![ji-eun_lee](https://user-images.githubusercontent.com/114035408/234686208-8f64554a-ca83-42fc-b194-bc04fb205c45.jpg)



* Dataset
  * [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset
    * [Images](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADSNUu0bseoCKuxuI5ZeTl1a/Img?dl=0&preview=img_align_celeba.zip) should be placed in `./data/img_align_celeba/*.jpg`
    * [Attribute labels](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAA8YmAHNNU6BEfWMPMfM6r9a/Anno?dl=0&preview=list_attr_celeba.txt) should be placed in `./data/list_attr_celeba.txt`
  * [HD-CelebA](https://github.com/LynnHo/HD-CelebA-Cropper) (optional)
    * Please see [here](https://github.com/LynnHo/HD-CelebA-Cropper).
  * [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans) dataset (optional)
    * Please see [here](https://github.com/willylulu/celeba-hq-modified).
    * _Images_ should be placed in `./data/celeba-hq/celeba-*/*.jpg`
    * _Image list_ should be placed in `./data/image_list.txt`
* [Pretrained models](http://bit.ly/attgan-pretrain): download the models you need and unzip the files to `./output/` as below,
  ```text
  output
  ├── 128_shortcut1_inject0_none
  ├── 128_shortcut1_inject1_none
  ├── 256_shortcut1_inject0_none
  ├── 256_shortcut1_inject1_none
  ├── 256_shortcut1_inject0_none_hq
  ├── 256_shortcut1_inject1_none_hq
  ├── 384_shortcut1_inject0_none_hq
  └── 384_shortcut1_inject1_none_hq
  ```


**Our Code is modified from the following Git hub**

*CLAM architecture https://github.com/mahmoodlab/CLAM*

*CycleGAN architecture https://github.com/ayukat1016/gan_sample/tree/main/chapter5*

## CLAM with Annotation 

### Pre-requisites:
* Linux (Tested on Ubuntu 20.04)
* NVIDIA GPU (Tested on Nvidia A100)
* Python (3.7.5), h5py (2.10.0), matplotlib (3.1.1), numpy (1.17.3), opencv-python (4.1.1.26), openslide-python (1.1.1), openslide (3.4.1), pandas (0.25.3), pillow (6.2.1), PyTorch (1.3.1), scikit-learn (0.22.1), scipy (1.3.1), tensorflow (1.14.0), tensorboardx (1.9), torchvision (0.4.2), smooth-topk.


### Prepare
1. Make annotation using QuPath [QuPath](https://qupath.github.io/)  
2. Export Object Data as geojson (Pretty JSON)  
   Note: Check the feature geometry type is exported as Polygon (Our code does not support the Mutlipolygon)
3. Make folder "ANN_PATH" and add annotation data(.geojson) to the folder  
   Note: The file name of the annotation data should be the same as the WSI file.  
         (ex. slide_1.ndpi & slide_1.geojson)


### WSI Segmentation and Make Patch 

<img src="Segmentation1.png" width="200px" align="center" />
The first step focuses on segmenting the tissue.  
The segmentation of the annotated area on the slides.  
Place the digitized whole slide image data (formated .ndpi, .svs etc.) under a folder named DATA_DIRECTORY

```bash
DATA_DIRECTORY/
	├── slide_1.ndpi
	├── slide_2.ndpi
	:	
	└── slide_X.ndpi
```

#### Run
``` shell
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --step_size 256 --seg --patch --stitch  --ann_path  ANN_DIR

```

The above command segment the slide in DATA_DIRECTORY using annotated ROIs.
The result (masks, patches,stitches and process_list_autogen.csv) is created at the following folder structure under RESULTS_DIRECTORY

```bash
RESULTS_DIRECTORY/
	├── masks
    		├── slide_1.png
    		├── slide_2.png
		:	:
    		└── slide_x.png
	├── patches
    		├── slide_1.h5
    		├── slide_2.h5
    		:	:
		└── slide_x.h5
	├── stitches
    		├── slide_1.png
    		├── slide_2.png
		:	:
    		└── slide_x.png
	└── process_list_autogen.csv

```

Check the **stitches** folder.   
It contains downsampled visualizations of stitched tissue patches (one image per slide)

In our model, we do not use information of MASK.  
Mask is created based on the CLAM preset.   
We use annotation information instead of MASK information.

### Feature Extraction (GPU Example)
<img src="featureextraction.png" width="300px" align="center" />

Move "process_list_autogen.csv" (Created above under the RESULT_DIRECTORY) to main directory (current directory)

####RUN
```bash
CUDA_VISIBLE_DEVICES=0,1 python extract_features_fp.py --data_h5_dir DIR_TO_COORDS --data_slide_dir DATA_DIRECTORY --csv_path CSV_FILE_NAME --feat_dir FEATURES_DIRECTORY --batch_size 512 --slide_ext .ndpi
```

The result (h5_files and pt_files) is created at the following folder structure under FEATURES_DIRECTORY

```bash
FEATURES_DIRECTORY/
    ├── h5_files
            ├── slide_1.h5
            ├── slide_2.h5
	    :	   :
            └── slide_x.h5
    └── pt_files
            ├── slide_1.pt
            ├── slide_2.pt
	    :	   :
            └── slide_x.pt
```


### Create Datasets and Train-Val-Test split
1. Create folder named "tumor_vs_normal_resnet_features" under DATA_ROOT_DIR     
2. Move the folde created by feature extraction (h5_files & pt_files) to the folder "tumor_vs_normal_resnet_features"    

The data used for training and testing are expected to be organized as follows:
```bash
DATA_ROOT_DIR/
    ├──feature_extracted/
        ├── h5_files
                ├── slide_1.h5
                ├── slide_2.h5
		:	:
                └── slide_x.h5
        └── pt_files
                ├── slide_1.pt
                ├── slide_2.pt
		:	:
                └── slide_x.pt
```

3. Make a dictionary with slide name and label. Edit "tumor_vs_normal_dummy_clean.csv" at dataset_csv    
4. Run the command below (ex. Number of fold:10, train/val/test 80/10/10)

``` shell
python create_splits_seq.py --task task_1_tumor_vs_normal --seed 1 --label_frac 1.0 --k 10 　
```








For training, look under main.py:
```python 
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_feat_resnet'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            label_col = 'label',
                            ignore=[])
```
The user would need to pass:
* csv_path: the path to the dataset csv file
* data_dir: the path to saved .pt features
* label_dict: a dictionary that maps labels in the label column to numerical values
* label_col: name of the label column (optional, by default it's 'label')
* ignore: labels to ignore (optional, by default it's an empty list)

Finally, the user should add this specific 'task' specified by this dataset object in the --task arguments as shown below:

```python
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping'])
```


The script uses the **Generic_WSI_Classification_Dataset** Class for which the constructor expects the same arguments as 
**Generic_MIL_Dataset** (without the data_dir argument). For details, please refer to the dataset definition in **datasets/dataset_generic.py**

### GPU Training Example for Binary Positive vs. Negative Classification (e.g. Lymph Node Status)
``` shell
CUDA_VISIBLE_DEVICES=0 python main.py --drop_out --early_stopping --lr 2e-4 --k 10 --label_frac 0.5 --exp_code task_1_tumor_vs_normal_CLAM_50 --weighted_sample --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type clam_sb --log_data --data_root_dir DATA_ROOT_DIR
```

### Testing and Evaluation Script
User also has the option of using the evluation script to test the performances of trained models. Examples corresponding to the models trained above are provided below:
``` shell
CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --models_exp_code task_1_tumor_vs_normal_CLAM_50_s1 --save_exp_code task_1_tumor_vs_normal_CLAM_50_s1_cv --task task_1_tumor_vs_normal --model_type clam_sb --results_dir results --data_root_dir DATA_ROOT_DIR
```

``` shell
CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --models_exp_code task_2_tumor_subtyping_CLAM_50_s1 --save_exp_code task_2_tumor_subtyping_CLAM_50_s1_cv --task task_2_tumor_subtyping --model_type clam_sb --results_dir results --data_root_dir DATA_ROOT_DIR
```


Once again, for information on each commandline argument, see:
``` shell
python eval.py -h
```

By adding your own custom datasets into **eval.py** the same way as you do for **main.py**, you can also easily test trained models on independent test sets. 

### Heatmap Visualization
Heatmap visualization can be computed in bulk via **create_heatmaps.py** by filling out the config file and storing it in **/heatmaps/configs** and then running **create_heatmaps.py** with the --config NAME_OF_CONFIG_FILE flag. A demo template is included (**config_template.yaml**) for lung subtyping on two WSIs from the CPTAC. 
To run the demo (raw results are saved in **heatmaps/heatmap_raw_results** and final results are saved in **heatmaps/heatmap_production_results**):
``` shell
CUDA_VISIBLE_DEVICES=0,1 python create_heatmaps.py --config config_template.yaml
```
See **/heatmaps/configs/config_template.yaml** for explanations for each configurable option.


### Trained Model Checkpoints
For reproducability, all trained models used can be accessed [here](https://drive.google.com/drive/folders/1NZ82z0U_cexP6zkx1mRk-QeJyKWk4Q7z?usp=sharing).
The 3 main folders (**tcga_kidney_cv**, **tcga_cptac_lung_cv** and **camelyon_40x_cv**) correspond to models for RCC subtyping trained on the TCGA, for NSCLC subtyping trained on TCGA and CPTAC and for Lymph Node Metastasis (Breast) detection trained on Camelyon16+17 respectively. In each main folder, each subfolder corresponds to one set of 10-fold cross-validation experiments. For example, the subfolder tcga_kidney_cv_CLAM_50_s1 contains the 10 checkpoints corresponding to the 10 cross-validation folds for TCGA RCC subtyping, trained using CLAM with multi-attention branches using 50% of cases in the full training set. 

For reproducability, these models can be evaluated on data prepared by following the same pipeline described in the sections above by calling **eval.py** with the appropriate arguments that specify the model options (--dropout should be enabled and either --model_type clam_mb or --model_type mil should be set, for evaluation only, --subtyping flag does not make a difference) as well as where the model checkpoints (--results_dir and --models_exp_code) and data (--data_root_dir and --task) are stored.

### Examples

Please refer to our pre-print and [interactive demo](http://clam.mahmoodlab.org) for detailed results on three different problems and adaptability across data sources, imaging devices and tissue content. 

Visulize additional examples here: http://clam.mahmoodlab.org

## Issues
- Please report all issues on the public forum.

## License
© [Mahmood Lab](http://www.mahmoodlab.org) - This code is made available under the GPLv3 License and is available for non-commercial academic purposes.

## Funding
This work was funded by NIH NIGMS [R35GM138216](https://reporter.nih.gov/search/sWDcU5IfAUCabqoThQ26GQ/project-details/10029418).

## Reference
If you find our work useful in your research or if you use parts of this code please consider citing our [paper](https://www.nature.com/articles/s41551-020-00682-w):

Lu, M.Y., Williamson, D.F.K., Chen, T.Y. et al. Data-efficient and weakly supervised computational pathology on whole-slide images. Nat Biomed Eng 5, 555–570 (2021). https://doi.org/10.1038/s41551-020-00682-w

```
@article{lu2021data,
  title={Data-efficient and weakly supervised computational pathology on whole-slide images},
  author={Lu, Ming Y and Williamson, Drew FK and Chen, Tiffany Y and Chen, Richard J and Barbieri, Matteo and Mahmood, Faisal},
  journal={Nature Biomedical Engineering},
  volume={5},
  number={6},
  pages={555--570},
  year={2021},
  publisher={Nature Publishing Group}
}
```

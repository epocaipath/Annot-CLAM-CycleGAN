from glob import glob # annotationフォルダから読み込み用
import json # annotationファイル読み込み用
import math
import os
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom
import multiprocessing as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np
import openslide
from PIL import Image
import pdb
import h5py
import math
from wsi_core.wsi_utils import savePatchIter_bag_hdf5, initialize_hdf5_bag, coord_generator, save_hdf5, sample_indices, screen_coords, isBlackPatch, isWhitePatch, to_percentiles
import itertools
from wsi_core.util_classes import isInContourV1, isInContourV2, isInContourV3_Easy, isInContourV3_Hard, Contour_Checking_fn
from utils.file_utils import load_pkl, save_pkl

Image.MAX_IMAGE_PIXELS = 933120000

# create_patches_fp.pyや、
# create_heatmaps.py -> heatmap_utils.pyから呼び出される
# objectにはwsiファイルが入る。create_heatmaps.pyのスライド毎のfor文の中でheatmap_utils.py経由で呼ばれるので、順にwsiが読み込まれる。
class WholeSlideImage(object):
    def __init__(self, path):
        # チェック用12
        print("WholeSlideImage's path:",path)
        # チェック用12終わり

        """
        Args:
            path (str): fullpath to WSI file
        """

        # self.nameにTumorやNormalなどが入る。Windowsの場合はパスがそのまま残る
        self.name = ".".join(path.split("/")[-1].split('.')[:-1])
        # self.wsiに各スライドののopenslideが入る
        self.wsi = openslide.open_slide(path)
        # self.level_downsamplesに下の方にある_assertLevelDownsamples()メソッドの返り値[(1.0,1.0),(2.0,2.0),...,(256.0,256.0)]を入れる
        # ここでは、level_downsamplesが間違いないか確認している。
        self.level_downsamples = self._assertLevelDownsamples()
        # self.level_dimにwsiに応じて各ダウンサンプルレベルにおける
        # 幅×高さサイズのタプル((23040, 13824),(11520, 6912),(5760, 3456),(2880, 1728),(1440, 864),…,(90, 54))等を入れる。
        self.level_dim = self.wsi.level_dimensions
        # self.contours_tissueはNone
        self.contours_tissue = None
        # self.contours_tumorはNone
        self.contours_tumor = None
        # self.hdf5_fileはNone
        self.hdf5_file = None

    def getOpenSlide(self):
        return self.wsi

    def initXML(self, xml_path):
        def _createContour(coord_list):
            return np.array([[[int(float(coord.attributes['X'].value)), 
                               int(float(coord.attributes['Y'].value))]] for coord in coord_list], dtype = 'int32')

        xmldoc = minidom.parse(xml_path)
        annotations = [anno.getElementsByTagName('Coordinate') for anno in xmldoc.getElementsByTagName('Annotation')]
        self.contours_tumor  = [_createContour(coord_list) for coord_list in annotations]
        self.contours_tumor = sorted(self.contours_tumor, key=cv2.contourArea, reverse=True)

    def initTxt(self,annot_path):
        def _create_contours_from_dict(annot):
            all_cnts = []
            for idx, annot_group in enumerate(annot):
                contour_group = annot_group['coordinates']
                if annot_group['type'] == 'Polygon':
                    for idx, contour in enumerate(contour_group):
                        contour = np.array(contour).astype(np.int32).reshape(-1,1,2)
                        all_cnts.append(contour) 

                else:
                    for idx, sgmt_group in enumerate(contour_group):
                        contour = []
                        for sgmt in sgmt_group:
                            contour.extend(sgmt)
                        contour = np.array(contour).astype(np.int32).reshape(-1,1,2)    
                        all_cnts.append(contour) 

            return all_cnts
        
        with open(annot_path, "r") as f:
            annot = f.read()
            annot = eval(annot)
        self.contours_tumor  = _create_contours_from_dict(annot)
        self.contours_tumor = sorted(self.contours_tumor, key=cv2.contourArea, reverse=True)

    def initSegmentation(self, mask_file):
        # load segmentation results from pickle file
        import pickle
        asset_dict = load_pkl(mask_file)
        self.holes_tissue = asset_dict['holes']
        self.contours_tissue = asset_dict['tissue']

    def saveSegmentation(self, mask_file):
        # save segmentation results using pickle
        asset_dict = {'holes': self.holes_tissue, 'tissue': self.contours_tissue}
        save_pkl(mask_file, asset_dict)

    # create_patches.pyのsegmentメソッドから呼び出し
    def segmentTissue(self, seg_level=0, sthresh=20, sthresh_up = 255, mthresh=7, close = 0, use_otsu=False, 
                            filter_params={'a_t':100}, ref_patch_size=512, exclude_ids=[], keep_ids=[]):
        # チェック用6
        print("filter_params:",filter_params)
        # チェック用6終わり
        """
            Segment the tissue via HSV -> Median thresholding -> Binary threshold
        """
        
        def _filter_contours(contours, hierarchy, filter_params):
            """
                Filter contours by: area.
            """
            filtered = []
            # 前景の輪郭のindexを探す。
            # hierarchyの各輪郭は[最初の子index,親index]なので、[:,1] == -1ということは、
            # 親のいない輪郭ということになる。
            # hierarchy_1は、親のいない輪郭の配列(1次元)を1次元にして、
            # indexを輪郭数だけリスト化したもの。
            # flatnonzeroは配列を1次元にして、
            # 0かFalse以外(True等)の値があるインデックスのリスト。(元から1次元だと思う)
            # hierarchy_1 = [前景輪郭のindex1,2,3,...]
            # find indices of foreground contours (parent == -1)
            hierarchy_1 = np.flatnonzero(hierarchy[:,1] == -1)
            all_holes = []

            # 前景の輪郭(hierarchy_1)の数だけ繰り返し            
            # loop through foreground contour indices
            for cont_idx in hierarchy_1:
                # 輪郭の中で、前景の輪郭をcontに入れる
                # actual contour
                cont = contours[cont_idx]
                # holes = cont_idx番目の前景輪郭の子の輪郭(穴とする)のindex(複数もありうる)
                #         穴が複数あればholesにも複数入る。
                # indices of holes contained in this contour (children of parent contour)
                holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)

                # aは前景の輪郭面積
                # take contour area (includes holes)
                a = cv2.contourArea(cont)

                # hole_areas = 該当輪郭の穴の面積のリスト
                # calculate the contour area of each hole
                hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]

                # 実際の面積 ＝ 親の輪郭(前景)の面積 - (子の輪郭(穴)の面積の合計)
                # actual area of foreground contour region
                a = a - np.array(hole_areas).sum()
                # 実際の面積がゼロなら飛ばす
                if a == 0: continue
                # a_t(組織面積の閾値)よりも実際の面積が大きいなら、
                # １．輪郭のindexをfileredリストに追加
                # ２．穴のindexリストをall_holesリストに追加
                if tuple((filter_params['a_t'],)) < tuple((a,)): 
                    filtered.append(cont_idx)
                    all_holes.append(holes)

            # foregrounde_contours = 穴を引いた面積がa_tを超える輪郭(組織と見做す)
            foreground_contours = [contours[cont_idx] for cont_idx in filtered]
            
            hole_contours = []
            # 1枚のスライド中の、全ての組織内に見つかった穴のセット数だけ繰り返す
            # hole_idsは各穴のインデックスだが、
            # 同じ組織内に穴が2つ以上ある場合、hole_idsには2つのインデックスが入る。
            for hole_ids in all_holes:
                # unfiltered_holes = 同じ組織内にある穴の輪郭情報リスト(1つの場合もあるはず)
                unfiltered_holes = [contours[idx] for idx in hole_ids ]
                # reverse=True=>面積(contourArea)が大きい順に穴の輪郭を並べ替える
                unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
                # 大きい順に、穴の最大数(max_n_holes)だけ、穴の輪郭として残す。
                # unfilered_holesとunfiltered_holesに違いがあるので要注意
                # unfilered_holes = 大きい順に、穴の最大数以内の輪郭(例: max_n_holes=2の場合、1組織内の穴の内、大きい2つの輪郭だけを穴とする。)
                # take max_n_holes largest holes by area
                unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
                filtered_holes = []
                
                # 1組織の中で、穴と見做された輪郭の数だけ繰り返す
                # filter these holes
                for hole in unfilered_holes:
                    # 穴の大きさが、a_hよりも大きければ、filtered_holesリストに保管する。
                    if cv2.contourArea(hole) > filter_params['a_h']:
                        filtered_holes.append(hole)
                # 組織単位(親輪郭単位)のfiltered_holesを、
                # スライド単位のhole_contoursリストに追加する。
                hole_contours.append(filtered_holes)
            # 組織輪郭、穴輪郭を返す
            return foreground_contours, hole_contours
        
        # パラメータのseg_levelに合わせて、wsiを読み込む
        img = np.array(self.wsi.read_region((0,0), seg_level, self.level_dim[seg_level]))
        # img_hsv = HSVに変換したwsi
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
        # img_med = HSV変換後のwsiをメディアンフィルタにかけたもの
        # ※メディアンフィルタ：フィルターサイズ内を中央値で平均化するので、
        #                     ノイズ除去やぼかしに使用される。
        # mthreshが大きいとノイズ除去＋ぼかしの効果が強くなる。
        # フィルターサイズなので、奇数でないといけない
        img_med = cv2.medianBlur(img_hsv[:,:,1], mthresh)  # Apply median blurring
               
        # use_otsuがTrueの場合は、大津の二値化を実施
        # ※大津の二値化：ピクセル値ヒストグラムの、２つのピークの間に閾値を決定(双峰性が必須)
        # img_otsu=処理後のイメージ
        # Thresholding
        if use_otsu:
            # 0＝アルゴリズムが自動で決定するため、sthresh_up=画素の最大値(デフォルト：255)
            # sthresh_upはパラメータにはなく、この関数でデフォルトとして与えられる引数と思われる。
            _, img_otsu = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        else:
            # sthreshで指定した閾値で二値化する。(8に設定していた)
            _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)

        # モルフォロジークロージングを実施
        # ※白黒の前景領域(=細胞領域：白)の黒い穴を埋めるのに有効
        # close=4でpresetファイル(custom_segment_level.csv)に設定されている。(default=0)
        # 0の場合は実施されない。
        # Morphological closing
        if close > 0:
            # closeはカーネルサイズ
            kernel = np.ones((close, close), np.uint8)
            img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)                 

        # scale = seglevelに合わせた倍率のタプル(seg_level=0=>(1.0,1.0), 1=>(2.0,2.0) etc.)
        # ここでのself.level_downsamplesはこのクラスの_assertLevelDownsamplesメソッドの
        # 返り値であり、openslideのものとは別なので要注意。
        # 1.0 = 1/1 ,2.0 = 1/2, 4.0 = 1/4 ...
        scale = self.level_downsamples[seg_level]
        # scaleは(2.0,2.0)となる。
#        print("scale:",scale)
        # このメソッドにて、ref_patch_size=512で設定されている。(パラメータではない)
        # scaled_ref_patch_area = (512 x 512) / (2.0 x 2.0) = 65536

        # 今回の例だと、a_tとa_hが65536倍に更新される。
        scaled_ref_patch_area = int(ref_patch_size**2 / (scale[0] * scale[1]))
        filter_params = filter_params.copy()
        filter_params['a_t'] = filter_params['a_t'] * scaled_ref_patch_area
        filter_params['a_h'] = filter_params['a_h'] * scaled_ref_patch_area
#        print("revised_filter_params:\n",filter_params)
        
        # Find and filter contours
        # contoursは輪郭情報(点数、1,2)のリスト(複数の輪郭が入っている)
        # hierarchyには、各輪郭について配列が入っている([次輪郭のindex,前輪郭のindex,最初の子のindex,親のindex])
        # cv2.RETR_CCOMPは輪郭抽出モード,
        # RETR_EXTERNAL=親だけ検出
        # RETR_LIST    =親として全検出
        # RETR_CCOMP   =2つの階層に分類,階層1,2のみ,2の子階層 = 1の孫階層は再び1となる。
        # RETR_TREE    =全ての階層構造を保持
        contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # Find contours 
        # hierarchyは3次元だが、1軸、サイズ1の軸がある(例: [[[1,-1,-1,-1],[2,0,-1,-1],[-1,1,-1,-1]]])
        # なのでnp.squeezeでサイズ1の0次元目を削除、輪郭の[最初の子index,親index]だけを抽出して
        # hierarchyに入れる。輪郭順に入っているはず。
        # hierarchy = [[最初の子index,親index],[最初の子index,親index]...]
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
        # filter_paramsがある場合、
        # _filter_contoursメソッドを呼び出し処理,以下が返ってくる
        # foreground_contours = 組織の輪郭情報
        # hole_contours = 穴の輪郭情報(例：[[[<組織1の穴1>点数,1,2],[<組織1の穴2>点数,1,2]],[<組織2の穴1>点数,1,2],...],..])
        if filter_params: foreground_contours, hole_contours = _filter_contours(contours, hierarchy, filter_params)  # Necessary for filtering out artifacts
        # チェック用7(組織の数と、組織ごとの穴の数をチェック)
        print("len(foreground_contours):",len(foreground_contours),"len(hole_contours):",len(hole_contours))
        # チェック用7終わり

        # scaleContourDimメソッドを呼び出す。(引数：組織の輪郭)
        # ダウンサンプリングのレベルに合わせて、輪郭情報を更新
        self.contours_tissue = self.scaleContourDim(foreground_contours, scale)
        # scaleHoleDimメソッドを呼び出す。(引数：穴の輪郭)
        # ダウンサンプリングのレベルに合わせて、輪郭情報を更新
        self.holes_tissue = self.scaleHolesDim(hole_contours, scale)

        # keep_idsが1つ以上入っている場合、keep_ids - exclude_idsの残り(set形式)が
        # contour_idsとなる
        # デフォルトではkeep_idsもexclude_idsも項目なし
        #exclude_ids = [0,7,9]
        if len(keep_ids) > 0:
            contour_ids = set(keep_ids) - set(exclude_ids)
        # keep_idsがない場合、輪郭の数(n)だけ0~nのリストを作成し、exclude_idsを引いて
        # contour_idsとしている。
        # デフォルトでは、keep_idsもexclude_idsもないので特に変更は無いと思われる。
        else:
            contour_ids = set(np.arange(len(self.contours_tissue))) - set(exclude_ids)

        # contours_tissure = contour_idsの数だけ組織の輪郭情報を入れたリスト
        self.contours_tissue = [self.contours_tissue[i] for i in contour_ids]
        # holes_tissure = contour_idsの数だけ穴の輪郭情報を入れたリスト
        self.holes_tissue = [self.holes_tissue[i] for i in contour_ids]

        print("contour_ids:",contour_ids)
        print("contours_tissue数:",len(self.contours_tissue))
        print("holes_tissue数:",len(self.holes_tissue))

    def visWSI(self, vis_level=0, color = (0,255,0), hole_color = (0,0,255), annot_color=(255,0,0), 
                    line_thickness=250, max_size=None, top_left=None, bot_right=None, custom_downsample=1, view_slide_only=False,
                    number_contours=False, seg_display=True, annot_display=True):
        
        downsample = self.level_downsamples[vis_level]
        scale = [1/downsample[0], 1/downsample[1]]
        
        if top_left is not None and bot_right is not None:
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
            region_size = (w, h)
        else:
            top_left = (0,0)
            region_size = self.level_dim[vis_level]

        img = np.array(self.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))
        
        if not view_slide_only:
            offset = tuple(-(np.array(top_left) * scale).astype(int))
            line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
            if self.contours_tissue is not None and seg_display:
                if not number_contours:
                    cv2.drawContours(img, self.scaleContourDim(self.contours_tissue, scale), 
                                     -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)

                else: # add numbering to each contour
                    for idx, cont in enumerate(self.contours_tissue):
                        contour = np.array(self.scaleContourDim(cont, scale))
                        M = cv2.moments(contour)
                        cX = int(M["m10"] / (M["m00"] + 1e-9))
                        cY = int(M["m01"] / (M["m00"] + 1e-9))
                        # draw the contour and put text next to center
                        cv2.drawContours(img,  [contour], -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)
                        cv2.putText(img, "{}".format(idx), (cX, cY),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 10)

                for holes in self.holes_tissue:
                    cv2.drawContours(img, self.scaleContourDim(holes, scale), 
                                     -1, hole_color, line_thickness, lineType=cv2.LINE_8)
            
            if self.contours_tumor is not None and annot_display:
                cv2.drawContours(img, self.scaleContourDim(self.contours_tumor, scale), 
                                 -1, annot_color, line_thickness, lineType=cv2.LINE_8, offset=offset)
        
        img = Image.fromarray(img)
    
        w, h = img.size
        if custom_downsample > 1:
            img = img.resize((int(w/custom_downsample), int(h/custom_downsample)))

        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size/w if w > h else max_size/h
            img = img.resize((int(w*resizeFactor), int(h*resizeFactor)))
       
        return img


    def createPatches_bag_hdf5(self, save_path, patch_level=0, patch_size=256, step_size=256, save_coord=True, **kwargs):
        contours = self.contours_tissue
        contour_holes = self.holes_tissue

        print("Creating patches for: ", self.name, "...",)
        elapsed = time.time()
        for idx, cont in enumerate(contours):
            patch_gen = self._getPatchGenerator(cont, idx, patch_level, save_path, patch_size, step_size, **kwargs)
            
            if self.hdf5_file is None:
                try:
                    first_patch = next(patch_gen)

                # empty contour, continue
                except StopIteration:
                    continue

                file_path = initialize_hdf5_bag(first_patch, save_coord=save_coord)
                self.hdf5_file = file_path

            for patch in patch_gen:
                savePatchIter_bag_hdf5(patch)

        return self.hdf5_file


    def _getPatchGenerator(self, cont, cont_idx, patch_level, save_path, patch_size=256, step_size=256, custom_downsample=1,
        white_black=True, white_thresh=15, black_thresh=50, contour_fn='four_pt', use_padding=True):
        start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1])
        print("Bounding Box:", start_x, start_y, w, h)
        print("Contour Area:", cv2.contourArea(cont))
        
        if custom_downsample > 1:
            assert custom_downsample == 2 
            target_patch_size = patch_size
            patch_size = target_patch_size * 2
            step_size = step_size * 2
            print("Custom Downsample: {}, Patching at {} x {}, But Final Patch Size is {} x {}".format(custom_downsample, patch_size, patch_size, 
                target_patch_size, target_patch_size))

        patch_downsample = (int(self.level_downsamples[patch_level][0]), int(self.level_downsamples[patch_level][1]))
        ref_patch_size = (patch_size*patch_downsample[0], patch_size*patch_downsample[1])
        
        step_size_x = step_size * patch_downsample[0]
        step_size_y = step_size * patch_downsample[1]
        
        if isinstance(contour_fn, str):
            if contour_fn == 'four_pt':
                cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'four_pt_hard':
                cont_check_fn = isInContourV3_Hard(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'center':
                cont_check_fn = isInContourV2(contour=cont, patch_size=ref_patch_size[0])
            elif contour_fn == 'basic':
                cont_check_fn = isInContourV1(contour=cont)
            else:
                raise NotImplementedError
        else:
            assert isinstance(contour_fn, Contour_Checking_fn)
            cont_check_fn = contour_fn

        img_w, img_h = self.level_dim[0]
        if use_padding:
            stop_y = start_y+h
            stop_x = start_x+w
        else:
            stop_y = min(start_y+h, img_h-ref_patch_size[1])
            stop_x = min(start_x+w, img_w-ref_patch_size[0])

        count = 0
        for y in range(start_y, stop_y, step_size_y):
            for x in range(start_x, stop_x, step_size_x):

                if not self.isInContours(cont_check_fn, (x,y), self.holes_tissue[cont_idx], ref_patch_size[0]): #point not inside contour and its associated holes
                    continue    
                
                count+=1
                patch_PIL = self.wsi.read_region((x,y), patch_level, (patch_size, patch_size)).convert('RGB')
                if custom_downsample > 1:
                    patch_PIL = patch_PIL.resize((target_patch_size, target_patch_size))
                
                if white_black:
                    if isBlackPatch(np.array(patch_PIL), rgbThresh=black_thresh) or isWhitePatch(np.array(patch_PIL), satThresh=white_thresh): 
                        continue

                patch_info = {'x':x // (patch_downsample[0] * custom_downsample), 'y':y // (patch_downsample[1] * custom_downsample), 'cont_idx':cont_idx, 'patch_level':patch_level, 
                'downsample': self.level_downsamples[patch_level], 'downsampled_level_dim': tuple(np.array(self.level_dim[patch_level])//custom_downsample), 'level_dim': self.level_dim[patch_level],
                'patch_PIL':patch_PIL, 'name':self.name, 'save_path':save_path}

                yield patch_info

        
        print("patches extracted: {}".format(count))

    # 以下のisInContoursメソッドより呼び出し
    # パッチの中心点が、穴に入っているかどうかをチェック。
    # 入っている,又は,穴の輪郭線上にある場合は1を返す。
    # 入っていなければ0を返す。
    @staticmethod
    def isInHoles(holes, pt, patch_size):
        for hole in holes:
#            print("hole:",hole)
            if cv2.pointPolygonTest(hole, (pt[0]+patch_size/2, pt[1]+patch_size/2), False) > 0:
                return 1
        
        return 0
    # 新規作成、腫瘍アノテーションした部分を切り抜く
    @staticmethod
    def isInRoI(pt, patch_size, ann_path, slide_name):
        # スライド名を確実にファイル名だけにする
        slide_name = os.path.basename(slide_name)
#        print("slide_name:",slide_name)
        '''
        ann_pathがフォルダでもファイルでも読み込む手順：
        ⓵ isfile,isdirでann_pathがファイルかフォルダか判定
        ⓶ ファイルの場合は以下の処理、フォルダの場合はglobでファイル名を取り出す。
        ⓷ フォルダの場合は、取り出したファイル名の内、設定条件(スライド名との完全一致orスライド名を含む)に合うファイルを取り出して、以下の処理を実施する。

        '''
        # 空の辞書を用意
        d = {}

        # ann_pathがファイルの場合 # 
        if os.path.isfile(ann_path):
            # ann_pathからjsonファイルを読み込む
            with open(ann_path) as f:
                d = json.load(f)
            # アノテーション座標データを入れるための空のリストを用意
            rois = []
            # qupathの仕様に合わせて、アノテーション座標を取り出し、アノテーション内にパッチ中心部があるかどうかチェック
            for fd in d['features']:
                if fd['geometry']['type'] == "Polygon":
                    rois.append(fd['geometry']['coordinates'])
                elif fd['geometry']['type'] == "MultiPolygon":
                    ro = fd['geometry']['coordinates']
                    for r in ro:
                        rois.append(r)
            for roi in rois:
                # リストroiを要素が整数(int32)のnp.arrayに変換する。
                if cv2.pointPolygonTest(np.array(roi).astype('int32'), (pt[0]+patch_size/2, pt[1]+patch_size/2), False) > 0:
                    # パッチ中心点がアノテーション領域内にあれば1を返す
                    return 1
            # パッチ中心点がアノテーション領域内になければ0を返す
            return 0

        # ann_pathがフォルダの場合 # 
        elif os.path.isdir(ann_path):
            ann_list = glob(ann_path+"/*json")
            for i in ann_list:
                ann_name = os.path.basename(i)
                # ann_nameにスライド名が含まれる条件
#                if slide_name in ann_name:
                # ann_name(拡張子なし)とスライド名が完全に一致する条件
                if slide_name == os.path.splitext(ann_name)[0]:
                    # ann_pathからjsonファイルを読み込む
                    with open(i) as f:
                        d = json.load(f)
                        
                    # アノテーション座標データを入れるための空のリストを用意
                    rois = []
                    
                    # qupathの仕様に合わせて、アノテーション座標を取り出し、アノテーション内にパッチ中心部があるかどうかチェック
                    for fd in d['features']:
                        if fd['geometry']['type'] == "Polygon":
                            rois.append(fd['geometry']['coordinates'])
                        elif fd['geometry']['type'] == "MultiPolygon":
                            ro = fd['geometry']['coordinates']
                            for r in ro:
                                rois.append(r)
                    for roi in rois:
                        # リストroiを要素が整数(int32)のnp.arrayに変換する。
                        if cv2.pointPolygonTest(np.array(roi).astype('int'), (pt[0]+patch_size/2, pt[1]+patch_size/2), False) > 0:
                            # パッチ中心点がアノテーション領域内にあれば1を返す
                            return 1
                    # パッチ中心点がアノテーション領域内になければ0を返す
                    return 0


    # process_coord_candidateメソッドから呼び出される。
    # pt=パッチのstart座標[x,y]
    @staticmethod
    def isInContours(cont_check_fn, pt, holes=None, patch_size=256, ann_path=None, slide_name=None): # パス追加 # スライド名追加
#        print("ann_path:",ann_path)
        # cont_check_fnインスタンスのクラスの__call__メソッド(pt)を呼び出している。
        # 組織の輪郭内にパッチ開始点が入っている場合は、以下をチェック
        if cont_check_fn(pt):
            # 穴の輪郭が存在する場合(テストファイルでは存在する)
            if holes is not None:
                # パッチの中心点が穴に入っていない場合
#                return not WholeSlideImage.isInHoles(holes, pt, patch_size)
                if not WholeSlideImage.isInHoles(holes, pt, patch_size):
                    # ann_pathがある場合
                    if ann_path is not None:
                        # roiに入っていたらTrue、入っていなかったらFalseを返す
                        return WholeSlideImage.isInRoI(pt, patch_size, ann_path, slide_name)
                    # ann_pathがなければ、Trueを返す
                    else:
                        return 1
                # 穴に入っていたらFalseを返す
                else:
                    return 0
            # 穴がない場合
            else:
                # 穴が無くて、アノテーションがある場合
                if ann_path is not None:
                    # roiに入っていたらTrue、 入っていなかったらFalseを返す
                    return WholeSlideImage.isInRoI(pt, patch_size, ann_path, slide_name)
                # ann_pathがなければ、Trueを返す
                else:
                    return 1

        # 組織外にパッチ開始点が入っている場合は、0を返す
        return 0
    
    # 組織の輪郭情報 × scaleで計算
    @staticmethod
    def scaleContourDim(contours, scale):
        return [np.array(cont * scale, dtype='int32') for cont in contours]

    # 穴の輪郭情報 × scaleで計算 (穴のリストは二重なので、scaleContourDimとは別に実施)
    @staticmethod
    def scaleHolesDim(contours, scale):
        return [[np.array(hole * scale, dtype = 'int32') for hole in holes] for holes in contours]

    # 同WholeSlideImageクラスの上部initから呼び出される
    def _assertLevelDownsamples(self):
        # 最初はlevel_downsamplesを空リストにする
        level_downsamples = []
        # 最大解像度(レベル0)でのスライドの(幅,高さ)のタプルをdim_0に入れる
        dim_0 = self.wsi.level_dimensions[0]
        
        # ダウンサンプルの全倍率のタプル(self.wsi.level_downsamples)と、ダウンサンプルレベル毎の全(幅,高さ)をそれぞれzipでセットにする。
        # セットの数だけ繰り返し、for文内でdownsample,dimとしてそれぞれ使えるようにする。(例:1回目downsample=1.0,dim=(23040, 13824))
        for downsample, dim in zip(self.wsi.level_downsamples, self.wsi.level_dimensions):
            # 幅と高さそれぞれのダウンサンプルレベルのタプルをestimated_downsampleに入れる。(0の次元の幅/dimの次元の幅,0の次元の高さ/dimの次元の高さ)
            estimated_downsample = (dim_0[0]/float(dim[0]), dim_0[1]/float(dim[1]))
            # (downsample,downsample)がestimated_downsampleと違う場合は、estimated_downsampleをlevel_downsamplesに入れる。
            # (downsample,downsample)がestimated_downsampleと同じ場合は、(downsample,downsample)をlevel_downsamplesに入れる。
            # 違うことがあるのか不明
            level_downsamples.append(estimated_downsample) if estimated_downsample != (downsample, downsample) else level_downsamples.append((downsample, downsample))        
        # [(1.0,1.0),(2.0,2.0),…(256.0,256.0)]のlevel_downsamplesを(恐らく上部のinitメソッドに)返す。
        return level_downsamples

    # create_patces.pyのpatchingメソッドなどから呼び出される。
    def process_contours(self, save_path, patch_level=0, patch_size=256, step_size=256, **kwargs):
        # チェック用8
        print("hdf5_save_path:",save_path)
        print("hdf5_name:",str(self.name))
        print("save_path_hdf5:",os.path.join(save_path, str(self.name)+'.h5'))
        # チェック用8終わり
        # パッチ作成後の.h5ファイル保存先のパスを作成する。
        # インプットフォルダをフルパスで実行した場合、self.name=インプットフォルダのフルパスなので、
        # 下のos.path.join(save_path, str(self.name)+".h5")にて、save_pathをインプットフォルダが上書きしてしまう。
        # そのため、h5ファイルがインプットフォルダに保存されてしまい、stitchingしてくれない。
        # これを予防するために、os.path.basenameでself.nameからファイル名だけを取り出す。
        # os.path.basename(self.name)
        save_path_hdf5 = os.path.join(save_path, str(os.path.basename(self.name)) + '.h5')
# 元コード1
#        save_path_hdf5 = os.path.join(save_path, str(self.name) + '.h5')
        print("Creating patches for: ", self.name, "...",)
        elapsed = time.time()
        # n_contours = セグメンテーション処理で得た組織の輪郭数
        n_contours = len(self.contours_tissue)
        print("Total number of contours to process: ", n_contours)
        # fp_chunk_size = 組織の輪郭数 * 0.05 で小数点以下を切り上げしたもの (例：math.ceil(1 * 0.05) = 1)
        # 輪郭数1~20 => 1 fp_chunk_size, 21~40 => 2 fp_chunk_size, ...
        fp_chunk_size = math.ceil(n_contours * 0.05)
        # init=Trueとする
        init = True
        # 輪郭の数だけ繰り返す
        for idx, cont in enumerate(self.contours_tissue):
            # 計算の条件を満たす場合、処理中の輪郭の番号と、輪郭数を表示する。
            # 少なくともfp_chunk_size==1,2の場合はFalseとなる。基本Falseになりそう
            if (idx + 1) % fp_chunk_size == fp_chunk_size:
                print('Processing contour {}/{}'.format(idx, n_contours))
            
            # asset_dict = パッチの開始点座標のリストがvalueに入った辞書
            # attr_dict = ファイル名やダウンサンプリングレベル等、
            # process_contourメソッドの実行返り値(同番号の穴輪郭、h5ファイル保存先、パッチレベル、サイズ、step_size(default==256)、その他係数を引数とする)
            # パッチのx,y座標の辞書
            asset_dict, attr_dict = self.process_contour(cont, self.holes_tissue[idx], patch_level, save_path, patch_size, step_size, **kwargs)
            
            # パッチの座標が1つでも記録されていたら、asset_dict,attr_dictをh5ファイルとして保存するため、
            # wsi_coreフォルダ内、wsi_utils.pyファイル内のsave_hdf5関数を呼び出す。
            if len(asset_dict) > 0:
                # init=Trueの場合、保存しinit=Falseにする。
                # つまり、最初の組織輪郭を保存する際はファイル名やパッチレベル等の情報(attr_dict)も一緒に保存するが、
                # ２つ目以降の組織輪郭に対しては、ファイル名等は同一で二重に保存する必要はないので、パッチ座標のみを保存する。
                # 恐らく、組織輪郭の情報ごとにh5ファイルを出力すると思われる。
                if init:
                    save_hdf5(save_path_hdf5, asset_dict, attr_dict, mode='w')
                    init = False
                # init＝Trueでない場合も、保存する
                else:
                    save_hdf5(save_path_hdf5, asset_dict, mode='a')
        print("self.hdf5_file:",self.hdf5_file)
        # create_patches_fp.pyのpatchingメソッドに返す。
        return self.hdf5_file


    # 上記process_contoursから呼び出される。patch_size,step_sizeはデフォルトで256
    # 『組織輪郭情報ごと』に呼び出され処理される。
    def process_contour(self, cont, contour_holes, patch_level, save_path, patch_size = 256, step_size = 256,
        contour_fn='four_pt', use_padding=True, top_left=None, bot_right=None, ann_path=None):
#        print("ann_path:",ann_path)
        # 輪郭情報(cont)がNoneでないなら、
        # start_x,start_y,w,h = 組織輪郭に外接する矩形の『左上x,左上y,幅,高さ』となる。

        # 輪郭情報がNoneなら(start_x=0,start_y=0,w=self.level_dim[patch_level][0],h=self.level_dim[patch_level][1])
        # ※self.level_dim[patch_level]は指定パッチレベルでの画像の『幅,高さ』のタプルである。
        # (例：self.level_dim[0] = (23040,13824)等)
        # つまり、輪郭情報がないときは、画像全体を示す。
        start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1])

        # patch_downsample = patch_levelに応じたダウンサンプルレベル
        # patch_level=0の場合:(1,1), 1の場合:(2,2)
        patch_downsample = (int(self.level_downsamples[patch_level][0]), int(self.level_downsamples[patch_level][1]))
        # patch_size =256(default), patch_downsample=(2,2) の場合：
        # (256 * 2, 256 * 2) = (512,512)
        ref_patch_size = (patch_size*patch_downsample[0], patch_size*patch_downsample[1])
        
        # パッチレベル[0]での画像の幅と高さ
        # 例：img_w = 23040, img_h = 13824
        img_w, img_h = self.level_dim[0]
        # パディングを用いる設定なら、
        # stop_y = 組織輪郭の外接矩形の右下y
        # stop_x = 組織輪郭の外接矩形の右下x
        if use_padding:
            stop_y = start_y+h
            stop_x = start_x+w
        else:
        # パディングを用いないなら、
        # stop_y = (『組織輪郭の外接矩形の右下y』 or 『画像高さ - パッチダウンサンプルレベルに合わせたパッチサイズ+1』の小さい方)
        # stop_x = (『組織輪郭の外接矩形の右下x』 or 『画像幅 - パッチダウンサンプルレベルに合わせたパッチサイズ+1』の小さい方)
            stop_y = min(start_y+h, img_h-ref_patch_size[1]+1)
            stop_x = min(start_x+w, img_w-ref_patch_size[0]+1)
        
        # バウンティングボックスとして、左上x,y,幅,高さを表示する。
        print("Bounding Box:", start_x, start_y, w, h)
        # 輪郭面積として、輪郭のcontourAreaで面積を表示
        print("Contour Area:", cv2.contourArea(cont))

        # bot_right,top_left = デフォルトでNone
        # それぞれ、対象領域を手動で決める場合に設定すると思われる。
        # Noneでない場合、stop_y = bot_right[1]か上記stop_yの内、小さい方
        # Noneでない場合、stop_x = bot_right[0]か上記stop_xの内、小さい方
        if bot_right is not None:
            stop_y = min(bot_right[1], stop_y)
            stop_x = min(bot_right[0], stop_x)
        # Noneでない場合、start_y = top_left[1]か上記start_yの大きい方
        # Noneでない場合、start_x = top_left[0]か上記start_xの大きい方
        if top_left is not None:
            start_y = max(top_left[1], start_y)
            start_x = max(top_left[0], start_x)

        # 分析対象領域を設定している場合、
        # bot_rightがNoneでない、又は、top_leftがNoneでない場合
        if bot_right is not None or top_left is not None:
            # w = stop_x - start_x, h = stop_y - start_y
            w, h = stop_x - start_x, stop_y - start_y
            # wが0以下の場合、又は、hが0以下の場合,
            # 『本輪郭は対象領域外、スキップ』と表示される。
            if w <= 0 or h <= 0:
                print("Contour is not in specified ROI, skip")
                return {}, {}
            # wやhが正の場合、
            # 『調整されたバウンティングボックス：start_x, start_y, w, h』と表示される。
            else:
                print("Adjusted Bounding Box:", start_x, start_y, w, h)
    
        # contour_fnが文字列の場合、各指定に応じてクラスを呼び出し
        if isinstance(contour_fn, str):
            # contour_fn == 'four_pt'の場合 ※デフォルト
            if contour_fn == 'four_pt':
                # cont_check_fn = wsi_coreフォルダ内、util_classes.py内のisInCountourV3_Easyクラスのインスタンス
                # 引数：組織輪郭、ダウンサンプリングレベルを考慮したパッチサイズ、中央シフト=0.5
                cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'four_pt_hard':
                cont_check_fn = isInContourV3_Hard(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'center':
                cont_check_fn = isInContourV2(contour=cont, patch_size=ref_patch_size[0])
            elif contour_fn == 'basic':
                cont_check_fn = isInContourV1(contour=cont)
            else:
                raise NotImplementedError
        else:
            # contour_fnが文字列でない場合は、Contour_Checking_fnのインスタンスでなければならない
            # その場合、cont_check_fn = contour_fnそのもの
            # ※ Contour_Checking_fn = util_classes.py内の上記isInContourV3_Easy等の親クラス
            assert isinstance(contour_fn, Contour_Checking_fn)
            cont_check_fn = contour_fn

        # step_size_x,y => パッチのダウンサンプリングレベルに合わせたstepサイズにする
        # step_size_x = step_size(default=256) * patch_downsample[0](=1.0 or 2.0等)
        step_size_x = step_size * patch_downsample[0]
        step_size_y = step_size * patch_downsample[1]
        
        # x_range,y_range = 輪郭のバウンティングボックス(外接矩形)に
        #                   パッチを当てた時の、
        #                   各パッチのxのスタート地点、yのスタート地点(np.array)
        # x_range = array([start_x, start_x + step_size_x, start_x + 2*(step_size_x),...])
        # x_range = array([start_x, start_x + step_size_x, start_x + 2*(step_size_x),...])
        x_range = np.arange(start_x, stop_x, step=step_size_x)
        y_range = np.arange(start_y, stop_y, step=step_size_y)

        # np.meshgridでパッチサイズの格子列を生成する,indexing='ij'=>行列の順序でグリッド生成
        # つまり、x_coordsとy_coordsを組み合わせると、組織内各パッチのスタート地点を表すことができる。
        # x_coords例：array([[2104,2104,2104,...(y_rangeの要素数=17個)],[2616,2616,...],[3128,3128,...],..])
        # y_coords例: array([[1944,2456,2968,...(y_rangeの要素)],[(y_rangeの要素)],...(x_rangeの要素の個数だけ(y_rangeの要素))])
        x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')

        # coord_candidates = 各パッチの開始点の[x座標,y座標]が入った二重リスト
        coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()

        num_workers = mp.cpu_count()
        if num_workers > 4:
            num_workers = 4
        # pool = 同時に実行するプロセス数(default=4)
        pool = mp.Pool(num_workers)

        #process_coord_candidate用の引数として、スライド名を用意
        slide_name = self.name

        # iterable = 各パッチ開始点ごとに引数を与えたリスト
        # 引数:(パッチ開始点xy座標,当該組織の穴の輪郭情報,パッチサイズ(ダウンサンプリング考慮済),
        #      isInContourV3_Easy等のインスタンス)
        # coord_cancidates内の、各パッチ開始座標(coord)ごとに上記引数をリストに入れる。
        iterable = [(coord, contour_holes, ref_patch_size[0], cont_check_fn, ann_path, slide_name) for coord in coord_candidates]

        # 最新opencv対応
        # opencv4.5.1.48以前でない場合、iterableのcontour coodinatesをint形式に変換しなければエラーとなる。
        # 参考：https://stackoverflow.com/questions/67837617/cant-parse-pt-sequence-item-with-index-0-has-a-wrong-type-pointpolygon-er
        # 最新のopencvできちんと動くように、int形式に変換する

        # 座標をintに変換したiterableをiterable_newに入れる。
        iterable_new = []
        # 各パッチ座標ごとに読み込み
        # (i=[各パッチの[x,y],穴輪郭,パッチサイズ(ds済),isInContourV3等インスタンス)
        for i in iterable:
            # iterable_line = [整数[パッチstrat x座標,　y座標],穴輪郭,パッチサイズ,上記インスタンス]
            iterable_line = [np.uint32(i[0][0]).item(), np.uint32(i[0][1]).item()], i[1], i[2], i[3], i[4], i[5]
            iterable_new.append(iterable_line)
        # 同ファイル下方のprocess_coord_candidateメソッド(staticmethod)を呼び出す。
        # results = 4つのプロセスで並列処理して、組織の輪郭内であり、穴にも入っていないパッチのstart座標のリスト
        # ※全てのパッチに対して、process_coord_candidateを実行し、返り値をリスト化してresultsに入れている点に注意。
#        print("ann_path:",iterable_new[0][4])
        results = pool.starmap(WholeSlideImage.process_coord_candidate, iterable_new)
        # 最新opencv対応終わり
        
        # 上記最新opencv対応の為に下記resultをコメントにする
#        results = pool.starmap(WholeSlideImage.process_coord_candidate, iterable)
        pool.close()
        # np.array配列に、組織の内部にあり、穴にもかかっていない実際の組織部分を表す各パッチの開始座標(x,y)を入れる。
        results = np.array([result for result in results if result is not None])
        
        # 組織内で、穴に掛かっていないパッチの数を表示
        print('Extracted {} coordinates'.format(len(results)))
        
        # 少なくとも1つ以上、組織輪郭内で穴にも掛からないパッチがある場合、
        if len(results)>1:
            # asset_dictを辞書として設定(key = 'coords, value = results')する。
            asset_dict = {'coords' :          results}
            # また、attrを辞書として設定(key = パッチのサイズ,指定パッチレベル,指定ダウンサンプル(1.0,2.0 etc.),
            #                                指定パッチレベルでの画像サイズを幅、高さでnp配列のタプルにしたもの,
            #                                指定パッチレベルでの画像サイズ,ファイル名,保存先パス)
            attr = {'patch_size' :            patch_size, # To be considered...
                    'patch_level' :           patch_level,
                    'downsample':             self.level_downsamples[patch_level],
                    'downsampled_level_dim' : tuple(np.array(self.level_dim[patch_level])),
                    'level_dim':              self.level_dim[patch_level],
                    'name':                   self.name,
                    'save_path':              save_path}
            # attr_dictを二重辞書として設定 (key = 'coords', value = 上記attr辞書)
            attr_dict = { 'coords' : attr}

            # asset_dict：組織内に該当するパッチのstart x,y座標リスト辞書(二重リスト？)
            # attr_dict:パッチサイズやレベル等の全体情報辞書
            # の２つを返す
            return asset_dict, attr_dict
        # 組織内のパッチが1つもない場合は空白で返す。
        else:
            return {}, {}

    # selfを用いないstaticmethod。
    # 同クラスのprocess_contourメソッドから呼び出され、同isInContours(staticmethod)メソッドを呼び出す。
    @staticmethod
    def process_coord_candidate(coord, contour_holes, ref_patch_size, cont_check_fn, ann_path=None, slide_name=None):
#        print("ann_path_proc:",ann_path)

        # パッチが組織の輪郭内に入っており、中心点が穴に入っていない場合は、パッチのx,y座標を返す
        if WholeSlideImage.isInContours(cont_check_fn, coord, contour_holes, ref_patch_size, ann_path, slide_name):
            return coord
        else:
            return None

    def visHeatmap(self, scores, coords, vis_level=-1, 
                   top_left=None, bot_right=None,
                   patch_size=(256, 256), 
                   blank_canvas=False, canvas_color=(220, 20, 50), alpha=0.4, 
                   blur=False, overlap=0.0, 
                   segment=True, use_holes=True,
                   convert_to_percentiles=False, 
                   binarize=False, thresh=0.5,
                   max_size=None,
                   custom_downsample = 1,
                   cmap='coolwarm'):

        """
        Args:
            scores (numpy array of float): Attention scores 
            coords (numpy array of int, n_patches x 2): Corresponding coordinates (relative to lvl 0)
            vis_level (int): WSI pyramid level to visualize
            patch_size (tuple of int): Patch dimensions (relative to lvl 0)
            blank_canvas (bool): Whether to use a blank canvas to draw the heatmap (vs. using the original slide)
            canvas_color (tuple of uint8): Canvas color
            alpha (float [0, 1]): blending coefficient for overlaying heatmap onto original slide
            blur (bool): apply gaussian blurring
            overlap (float [0 1]): percentage of overlap between neighboring patches (only affect radius of blurring)
            segment (bool): whether to use tissue segmentation contour (must have already called self.segmentTissue such that 
                            self.contours_tissue and self.holes_tissue are not None
            use_holes (bool): whether to also clip out detected tissue cavities (only in effect when segment == True)
            convert_to_percentiles (bool): whether to convert attention scores to percentiles
            binarize (bool): only display patches > threshold
            threshold (float): binarization threshold
            max_size (int): Maximum canvas size (clip if goes over)
            custom_downsample (int): additionally downscale the heatmap by specified factor
            cmap (str): name of matplotlib colormap to use
        """

        if vis_level < 0:
            vis_level = self.wsi.get_best_level_for_downsample(32)

        downsample = self.level_downsamples[vis_level]
        scale = [1/downsample[0], 1/downsample[1]] # Scaling from 0 to desired level
                
        if len(scores.shape) == 2:
            scores = scores.flatten()

        if binarize:
            if thresh < 0:
                threshold = 1.0/len(scores)
                
            else:
                threshold =  thresh
        
        else:
            threshold = 0.0

        ##### calculate size of heatmap and filter coordinates/scores outside specified bbox region #####
        if top_left is not None and bot_right is not None:
            scores, coords = screen_coords(scores, coords, top_left, bot_right)
            coords = coords - top_left
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
            region_size = (w, h)

        else:
            region_size = self.level_dim[vis_level]
            top_left = (0,0)
            bot_right = self.level_dim[0]
            w, h = region_size

        patch_size  = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
        coords = np.ceil(coords * np.array(scale)).astype(int)
        
        print('\ncreating heatmap for: ')
        print('top_left: ', top_left, 'bot_right: ', bot_right)
        print('w: {}, h: {}'.format(w, h))
        print('scaled patch size: ', patch_size)

        ###### normalize filtered scores ######
        if convert_to_percentiles:
            scores = to_percentiles(scores) 

        scores /= 100
        
        ######## calculate the heatmap of raw attention scores (before colormap) 
        # by accumulating scores over overlapped regions ######
        
        # heatmap overlay: tracks attention score over each pixel of heatmap
        # overlay counter: tracks how many times attention score is accumulated over each pixel of heatmap
        overlay = np.full(np.flip(region_size), 0).astype(float)
        counter = np.full(np.flip(region_size), 0).astype(np.uint16)      
        count = 0
        for idx in range(len(coords)):
            score = scores[idx]
            coord = coords[idx]
            if score >= threshold:
                if binarize:
                    score=1.0
                    count+=1
            else:
                score=0.0
            # accumulate attention
            overlay[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] += score
            # accumulate counter
            counter[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] += 1

        if binarize:
            print('\nbinarized tiles based on cutoff of {}'.format(threshold))
            print('identified {}/{} patches as positive'.format(count, len(coords)))
        
        # fetch attended region and average accumulated attention
        zero_mask = counter == 0

        if binarize:
            overlay[~zero_mask] = np.around(overlay[~zero_mask] / counter[~zero_mask])
        else:
            overlay[~zero_mask] = overlay[~zero_mask] / counter[~zero_mask]
        del counter 
        if blur:
            overlay = cv2.GaussianBlur(overlay,tuple((patch_size * (1-overlap)).astype(int) * 2 +1),0)  

        if segment:
            tissue_mask = self.get_seg_mask(region_size, scale, use_holes=use_holes, offset=tuple(top_left))
            # return Image.fromarray(tissue_mask) # tissue mask
        
        if not blank_canvas:
            # downsample original image and use as canvas
            img = np.array(self.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))
        else:
            # use blank canvas
            img = np.array(Image.new(size=region_size, mode="RGB", color=(255,255,255))) 

        #return Image.fromarray(img) #raw image

        print('\ncomputing heatmap image')
        print('total of {} patches'.format(len(coords)))
        twenty_percent_chunk = max(1, int(len(coords) * 0.2))

        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        
        for idx in range(len(coords)):
            if (idx + 1) % twenty_percent_chunk == 0:
                print('progress: {}/{}'.format(idx, len(coords)))
            
            score = scores[idx]
            coord = coords[idx]
            if score >= threshold:

                # 幅と0が入ったデータ
                # attention block
                raw_block = overlay[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]]
                
                # image block (either blank canvas or orig image)
                img_block = img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]].copy()

                # color block (cmap applied to attention block)
                color_block = (cmap(raw_block) * 255)[:,:,:3].astype(np.uint8)

                if segment:
                    # tissue mask block
                    mask_block = tissue_mask[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] 
                    # copy over only tissue masked portion of color block
                    img_block[mask_block] = color_block[mask_block]
                else:
                    # copy over entire color block
                    img_block = color_block

                # rewrite image block
                img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] = img_block.copy()
        
        #return Image.fromarray(img) #overlay
        print('Done')
        del overlay

        if blur:
            img = cv2.GaussianBlur(img,tuple((patch_size * (1-overlap)).astype(int) * 2 +1),0)  

        if alpha < 1.0:
            img = self.block_blending(img, vis_level, top_left, bot_right, alpha=alpha, blank_canvas=blank_canvas, block_size=1024)
        
        img = Image.fromarray(img)
        w, h = img.size

        if custom_downsample > 1:
            img = img.resize((int(w/custom_downsample), int(h/custom_downsample)))

        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size/w if w > h else max_size/h
            img = img.resize((int(w*resizeFactor), int(h*resizeFactor)))
       
        return img

    
    def block_blending(self, img, vis_level, top_left, bot_right, alpha=0.5, blank_canvas=False, block_size=1024):
        print('\ncomputing blend')
        downsample = self.level_downsamples[vis_level]
        w = img.shape[1]
        h = img.shape[0]
        block_size_x = min(block_size, w)
        block_size_y = min(block_size, h)
        print('using block size: {} x {}'.format(block_size_x, block_size_y))

        shift = top_left # amount shifted w.r.t. (0,0)
        for x_start in range(top_left[0], bot_right[0], block_size_x * int(downsample[0])):
            for y_start in range(top_left[1], bot_right[1], block_size_y * int(downsample[1])):
                #print(x_start, y_start)

                # 1. convert wsi coordinates to image coordinates via shift and scale
                x_start_img = int((x_start - shift[0]) / int(downsample[0]))
                y_start_img = int((y_start - shift[1]) / int(downsample[1]))
                
                # 2. compute end points of blend tile, careful not to go over the edge of the image
                y_end_img = min(h, y_start_img+block_size_y)
                x_end_img = min(w, x_start_img+block_size_x)

                if y_end_img == y_start_img or x_end_img == x_start_img:
                    continue
                #print('start_coord: {} end_coord: {}'.format((x_start_img, y_start_img), (x_end_img, y_end_img)))
                
                # 3. fetch blend block and size
                blend_block = img[y_start_img:y_end_img, x_start_img:x_end_img] 
                blend_block_size = (x_end_img-x_start_img, y_end_img-y_start_img)
                
                if not blank_canvas:
                    # 4. read actual wsi block as canvas block
                    pt = (x_start, y_start)
                    canvas = np.array(self.wsi.read_region(pt, vis_level, blend_block_size).convert("RGB"))     
                else:
                    # 4. OR create blank canvas block
                    canvas = np.array(Image.new(size=blend_block_size, mode="RGB", color=(255,255,255)))

                # 5. blend color block and canvas block
                img[y_start_img:y_end_img, x_start_img:x_end_img] = cv2.addWeighted(blend_block, alpha, canvas, 1 - alpha, 0, canvas)
        return img

    def get_seg_mask(self, region_size, scale, use_holes=False, offset=(0,0)):
        print('\ncomputing foreground tissue mask')
        tissue_mask = np.full(np.flip(region_size), 0).astype(np.uint8)
        contours_tissue = self.scaleContourDim(self.contours_tissue, scale)
        offset = tuple((np.array(offset) * np.array(scale) * -1).astype(np.int32))

        contours_holes = self.scaleHolesDim(self.holes_tissue, scale)
        contours_tissue, contours_holes = zip(*sorted(zip(contours_tissue, contours_holes), key=lambda x: cv2.contourArea(x[0]), reverse=True))
        for idx in range(len(contours_tissue)):
            cv2.drawContours(image=tissue_mask, contours=contours_tissue, contourIdx=idx, color=(1), offset=offset, thickness=-1)

            if use_holes:
                cv2.drawContours(image=tissue_mask, contours=contours_holes[idx], contourIdx=-1, color=(0), offset=offset, thickness=-1)
            # contours_holes = self._scaleContourDim(self.holes_tissue, scale, holes=True, area_thresh=area_thresh)
                
        tissue_mask = tissue_mask.astype(bool)
        print('detected {}/{} of region as tissue'.format(tissue_mask.sum(), tissue_mask.size))
        return tissue_mask





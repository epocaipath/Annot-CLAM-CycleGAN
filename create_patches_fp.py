# internal imports
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df
# other imports
import os
import numpy as np
import time
import argparse
import pdb
import pandas as pd

# seg_and_patch関数の中で、patchingの後で呼び出される
def stitching(file_path, wsi_object, downscale = 64):
	start = time.time()
	# .h5ファイルのfile_path、wsi、ダウンスケール等を引数に、wsi_coreフォルダ内のwsi_utils.pyファイル内のStitchCoords
	heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0,0,0), alpha=-1, draw_grid=False)
	total_time = time.time() - start
	
	return heatmap, total_time

# 以下seg_and_patches関数でcurrent_parameterを用意してから、segmentが呼び出される。
def segment(WSI_object, seg_params, filter_params):

	'''
	WholeSlideImageクラスのsegmentTissueメソッドを実行する
	➡セグメンテーションを行う
	セグメンテーション後のWSIと、実行時間を返す。

	1.指定レベルで読み込み
	2.HSV化
	3.メディアンフィルタ(ノイズ除去、ぼかし)
	4.二値化(大津 or 通常)
	5.モルフォロジークロージング(細胞部の穴埋め)
	6.findcontour
	(輪郭を決定し、子輪郭を穴として、親輪郭area-子輪郭area=実面積
	実面積が閾値を超える場合は「組織」とみなす。
	さらに、1組織内の穴は、大きい順にmax_n_holes以下のものが抽出され、
	更に'a_h'よりも大きなサイズの穴だけが「穴」として見なされる。)
	'''
	### Start Seg Timer
	start_time = time.time()

	# Segment
	WSI_object.segmentTissue(**seg_params, filter_params=filter_params)
	# チェック用11
	print("**seg_params:",seg_params)
	# チェック用11終わり

	### Stop Seg Timers
	seg_time_elapsed = time.time() - start_time   
	return WSI_object, seg_time_elapsed

# 以下seg_and_patches関数でパッチ用current_parameterを用意してから、patchingメソッドを呼び出す。
def patching(WSI_object, **kwargs):
	'''
	WholeSlideImageクラスのprocess_contoursメソッドを実行する
	➡組織のパッチ座標の作成を行う
	空のパスと、実行時間を返す。(ファイルは内部で.h5形式で保存)

	1.指定レベルとパッチサイズ(256位？)に合わせて、組織の輪郭内にあるパッチが始まる座標を取得
	2.中心部が穴の輪郭内にあるパッチが始まる座標を取得
	3.組織の輪郭内にあって、穴の輪郭内にはない、つまり組織を表すパッチが始まる座標(asset)のリストを辞書形式で作成
	4.ファイル特性情報のデータ(attr)も辞書形式で作成
	5.輪郭ごとにパッチ座標(asset)のファイル(.h5)を保存
	6.file_path(＝空)、処理時間を返す
	'''
	### Start Patch Timer
	start_time = time.time()

	# チェック用10
	print("**kwargs:",kwargs)
        # チェック用10終わり

    # パッチ用パラメータを引数として、各wsiに対してprocess_contoursメソッドを実行する。
	# Patch
	file_path = WSI_object.process_contours(**kwargs)

	### Stop Patch Timer
	patch_time_elapsed = time.time() - start_time
	# 空のファイルパスと処理時間を返す
	return file_path, patch_time_elapsed

'''
セグメンテーションとパッチ作業などを実行する主要関数
'''
# 後半の引数はデフォルト値が設定されており、値が設定されていなくてもデフォルト処理される
def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir, 
				  patch_size = 512, step_size = 512, 
				  seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'},
				  filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}, 
				  vis_params = {'vis_level': -1, 'line_thickness': 500},
				  patch_params = {'use_padding': True, 'contour_fn': 'four_pt'},
				  patch_level = 0,
				  use_default_params = False, 
				  seg = False, save_mask = True, 
				  stitch= False, 
				  patch = False, auto_skip=True, process_list = None, ann_path = None):

    # slides＝wsiフォルダにあるwsiファイル名をリストにして並べ替える
	slides = sorted(os.listdir(source))
	# slidesにディレクトリ等が含まれる場合は除外
	# slidesが実際にファイルであれば、改めてslidesリストに入れる
	slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]
    # process_list指定がない場合、wsi_coreフォルダ内、batch_process_utils.py内の
	# initialize_df関数を呼び出し各スライドに対するパラメータ値を記載したdfを作成する。
	if process_list is None:
		df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
	
    # process_listがある場合は、process_listファイル(.csv)を読み込み、改めてdfを作成する。
	else:
		df = pd.read_csv(process_list)
		df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

	# maskにはTrueかFalseが入る。
    # df['process']には各行(各スライド)に対し1の値が入っている。
	# 1が入っている数だけTrueとなる。
	mask = df['process'] == 1
    # process_stack = dfの内、Trueの行だけを集めたDFになる。
	# (スライドの数だけ行があり、各行にa_hなどの設定列がある。)
	process_stack = df[mask]
	# totalはTrueとFalseの数が入る。
	total = len(process_stack)

	# チェック用1
	print("process_list=>df:\n",df)
	print("mask:",mask)
	print("len(df(mask)):",len(df[mask]))
	print("df['process']:",df['process'])
	print("total:",total)
	print("process_stack:\n",process_stack)
	# チェック用1終わり

    # dfの列名に'a'という名前の列がある場合は、legacy_support=Trueとなる。
	# 基本はないはず(a_h,a_tなどになっている)なので置いておく。
	legacy_support = 'a' in df.keys()
	if legacy_support:
		print('detected legacy segmentation csv file, legacy support enabled')
		df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
		'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
		'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
		'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
		'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

	seg_times = 0.
	patch_times = 0.
	stitch_times = 0.

    # mask==Trueの数(スライドの数)だけ繰り返す
	for i in range(total):
		# スライド毎にa_h等を記載した設定のdfをprocess_list_autgen.csvとして出力する。
		df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
		
		# idx = df(process_stack)の行番号が順に入る
		idx = process_stack.index[i]
		# チェック用2
		print("idx:",idx)
		# チェック用2終わり
		# slide = idx行目のslide_id(スライドファイル名)が入る
		slide = process_stack.loc[idx, 'slide_id']
		# チェック用3
		print("slide:",slide)
		# チェック用3終わり
		print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i, total))
		print('processing {}'.format(slide))
		
		# idx行目のprocess列を0にする。＝＞処理が終わったというフラグ
		df.loc[idx, 'process'] = 0
		# slide_id = 拡張子をのぞいたファイル名
		slide_id, _ = os.path.splitext(slide)

        # デフォルトでauto_skip = True
		# auto_skip = Trueで、パッチ保存フォルダ内にidx行目のファイル.h5が存在する場合、
		# 既に保管先にファイルがあるとしてスキップする旨のメッセージが表示されて、
		# 次の行へ移動する。
		if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
			print('{} already exist in destination location, skipped'.format(slide_id))
			df.loc[idx, 'status'] = 'already_exist'
			continue

		# Inialize WSI
		# full_path = 入力ディレクトリとslideファイル名を結合した、入力ファイルのフルパス
		full_path = os.path.join(source, slide)
		# WSI_object = wsi_coreフォルダ内、WholeSlideImage.py内、WholeSlideImageクラスのインスタンス
		WSI_object = WholeSlideImage(full_path)
		# チェック用5
		print("full_path:",full_path)
		# チェック用5終わり
		

        # (基本False) use_default_paramsがTrueなら、current_...にパラメータを入れていく。
		if use_default_params:
			current_vis_params = vis_params.copy()
			current_filter_params = filter_params.copy()
			current_seg_params = seg_params.copy()
			current_patch_params = patch_params.copy()
			
		# Falseなら、各パラメータに空の辞書を用意する。
		else:
			print("use_default_params:",use_default_params)
			current_vis_params = {}
			current_filter_params = {}
			current_seg_params = {}
			current_patch_params = {}

			'''
			以降の行でcurrent_パラメータに値を入れる。
			その後、current_パラメータを用いてフィルター、マスク、パッチを行う。
			'''
            # vis_paramsのキー(vis_level, line_thickness)毎に見て、
			# legacy_supportがTrueで'vis_level'列の場合、dfのvis_level列を-1にする。
			# その後、current_vis_params辞書をkey:-1で更新する。
			# 基本はlegacy_supportがFalse
			# legacy_supportがFalseなら、現行の値をcurrent_vis_paramsに入れる
			for key in vis_params.keys():
				if legacy_support and key == 'vis_level':
					df.loc[idx, key] = -1
				current_vis_params.update({key: df.loc[idx, key]})
				
#			print("current_vis_params_fc1:",current_vis_params)

            # filter_paramsのキー(a_t,a_h,max_n_holes)毎に見て、
			# a_tについて、legacy_supportがTrueなら、'a'列の値等を読み取り計算する。
			# 基本はlegacy_supportがFalse
			# legacy_supportがFalseなら、現行の値をcurrent_filter_paramsに入れる
			for key in filter_params.keys():
				if legacy_support and key == 'a_t':
					old_area = df.loc[idx, 'a']
					seg_level = df.loc[idx, 'seg_level']
					scale = WSI_object.level_downsamples[seg_level]
					adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
					current_filter_params.update({key: adjusted_area})
					df.loc[idx, key] = adjusted_area
				current_filter_params.update({key: df.loc[idx, key]})

            # legacy_supportがTrueの場合、seg_levelを-1にする。
			# 基本はlegacy_supportがFalseなので関係ないはず
			for key in seg_params.keys():
				if legacy_support and key == 'seg_level':
					df.loc[idx, key] = -1
				current_seg_params.update({key: df.loc[idx, key]})

			for key in patch_params.keys():
				current_patch_params.update({key: df.loc[idx, key]})
		
		# current_vis_levelが0未満(-1)の場合、
		if current_vis_params['vis_level'] < 0:
		    # wsiのlevel_dimensions((23040,13824),(11520,6912),...)の長さが1の時、
		    # current_vis_paramのvis_levelを0にする。
		    # 基本長さ9のはずなので該当しない気がする
			if len(WSI_object.level_dim) == 1:
				current_vis_params['vis_level'] = 0
			
			# openslideで読み込んだwsiを、wsiに入れる。
			# best_level = サイズが1\64となるレベルが入る。(best_level = 6)
			# best_levelをcurrent_vis_paramsの'vis_level'に入れる
			else:	
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_vis_params['vis_level'] = best_level

        # current_seg_paramsのseg_levelが0未満(-1)の場合、
		if current_seg_params['seg_level'] < 0:
		    # wsiのlevel_dimensions((23040,13824),(11520,6912),...)の長さが1の時、
			# seg_levelを0にする。
			if len(WSI_object.level_dim) == 1:
				current_seg_params['seg_level'] = 0
			
			# openslideで読み込んだwsiを、wsiに入れる。
			# best_level = サイズが1\64となるレベルが入る。(best_level = 6)
			# best_levelをcurrent_seg_paramsの'seg_level'に入れる
			else:
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_seg_params['seg_level'] = best_level

		# keep_ids = keep_idsの設定値を文字列で入れる。(基本"none")
		keep_ids = str(current_seg_params['keep_ids'])
		# noneではなく、値も入っている場合(Trueの場合?)、
		if keep_ids != 'none' and len(keep_ids) > 0:
			str_ids = current_seg_params['keep_ids']
			current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['keep_ids'] = []

		exclude_ids = str(current_seg_params['exclude_ids'])
		if exclude_ids != 'none' and len(exclude_ids) > 0:
			str_ids = current_seg_params['exclude_ids']
			current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['exclude_ids'] = []
        
		# w, h = level_dim((23040,13824),(11520,6912),...)のseg_level番目の(幅,高さ)
		# seg_levelが-1なら、後ろからなので一番小さい値を取る事になる。
		w, h = WSI_object.level_dim[current_seg_params['seg_level']] 
		# 幅 x 高さ が100000000を超えたらエラーメッセージが出てそのスライドは処理を飛ばされる。
		# 幅 x 高さ が100000000 x 100000000を超える場合にエラーという事に変更した。
#		if w * h > 1e8:
		if w * h > 1e16:
			print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
			df.loc[idx, 'status'] = 'failed_seg'
			continue

		df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
		df.loc[idx, 'seg_level'] = current_seg_params['seg_level']
		
		print("current_vis_params:\n",current_vis_params)
		print("current_seg_params:\n",current_seg_params)
		print("current_filter_params:\n",current_filter_params)

		'''

		セグメンテーション・マスク・パッチを実施

		'''
        # seg_time_elapsed = -1とする。
		seg_time_elapsed = -1
        # defaultはfalseだが、コマンドに--segを付けて実行していたらTrue。
		# seg = Trueなら、segmentメソッドを実行する。(引数=wsiデータ,currentパラメータ)
		if seg:
			WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params) 

        # save_mask = Trueなら、visWSIメソッドを実行し、マスク用の枠を描いたファイル(.jpg)
		# を出力フォルダのmasksフォルダに保存する。
		if save_mask:
			mask = WSI_object.visWSI(**current_vis_params)
			mask_path = os.path.join(mask_save_dir, slide_id+'.jpg')
			mask.save(mask_path)
		
		# patch = Trueであれば、current_patchパラメータを更新して、上方のpatchingメソッドを呼び出す

		patch_time_elapsed = -1 # Default time
		if patch:
			current_patch_params.update({'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size, 
										 'save_path': patch_save_dir})
			# ann_path追加
			current_patch_params['ann_path'] = ann_path
			# チェック用9
			print("current_patch_params:",current_patch_params)
			# チェック用9終わり
			file_path, patch_time_elapsed = patching(WSI_object = WSI_object,  **current_patch_params,)
			# チェック用4g
			print("file_path_p",file_path)
			# この時点で、file_pathの中身はh5ファイルのパスではなく、空。
			# チェック用4g終わり
		
		
		# stitch = Trueであれば、patchフォルダ＋スライド名.h5パスを作成して、
		# ファイルがあることを確認し、パッチを結合するstitchingメソッドを呼び出す。
		stitch_time_elapsed = -1
		if stitch:
			# file_path=パッチ保存先パス(.h5)
			file_path = os.path.join(patch_save_dir, slide_id+'.h5')
			# チェック用4
			print("file_path",file_path)
			# チェック用4終わり
			# パスにファイルがあれば、ファイルパス(.h5)とWSIデータを元にstitchingメソッドを呼び出す。
			if os.path.isfile(file_path):
				# 画像に保存するときにダウンスケールしている(1/64?)
				heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
				# パッチファイルを元に繋ぎ合わせたjpgファイルをstitch_pathに保存する。
				stitch_path = os.path.join(stitch_save_dir, slide_id+'.jpg')
				'''
				heatmapにマスク画像を合成して、マスク部分だけを残す
				heatmapはnumpy形式と思われるので、
				'''
				heatmap.save(stitch_path)

		print("segmentation took {} seconds".format(seg_time_elapsed))
		print("patching took {} seconds".format(patch_time_elapsed))
		print("stitching took {} seconds".format(stitch_time_elapsed))
		df.loc[idx, 'status'] = 'processed'

		seg_times += seg_time_elapsed
		patch_times += patch_time_elapsed
		stitch_times += stitch_time_elapsed

	seg_times /= total
	patch_times /= total
	stitch_times /= total

	df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
	print("average segmentation time in s per slide: {}".format(seg_times))
	print("average patching time in s per slide: {}".format(patch_times))
	print("average stiching time in s per slide: {}".format(stitch_times))
		
	return seg_times, patch_times

parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type = str,
					help='path to folder containing raw wsi image files')
parser.add_argument('--step_size', type = int, default=256,
					help='step_size')
parser.add_argument('--patch_size', type = int, default=256,
					help='patch_size')
parser.add_argument('--patch', default=False, action='store_true')
parser.add_argument('--seg', default=False, action='store_true')
parser.add_argument('--stitch', default=False, action='store_true')
parser.add_argument('--no_auto_skip', default=True, action='store_false')
parser.add_argument('--save_dir', type = str,
					help='directory to save processed data')
parser.add_argument('--preset', default=None, type=str,
					help='predefined profile of default segmentation and filter parameters (.csv)')
parser.add_argument('--patch_level', type=int, default=0, 
					help='downsample level at which to patch')
parser.add_argument('--process_list',  type = str, default=None,
					help='name of list of images to process with parameters (.csv)')
parser.add_argument('--ann_path',  type = str, default=None,
					help='path to directory containing json files, or , a json file containing roi information (.json)')

if __name__ == '__main__':
	# 引数設定
	args = parser.parse_args()

    # patch_save_dir = 引数save_dir + "patches"
	patch_save_dir = os.path.join(args.save_dir, 'patches')
	# mask_save_dir = 引数save_dir + "masks"
	mask_save_dir = os.path.join(args.save_dir, 'masks')
	# stitch_save_dir = 引数save_dir + "stitches"
	stitch_save_dir = os.path.join(args.save_dir, 'stitches')

    # 引数process_listがある場合は、process_list = 引数save_dir + 引数process_list
	if args.process_list:
		process_list = os.path.join(args.save_dir, args.process_list)

	# process_listがない場合はNone
	else:
		process_list = None

    # source(入力ディレクトリ), patch_save_dir, mask_save_dir, stitch_save_dirを表示する。
	print('source: ', args.source)
	print('patch_save_dir: ', patch_save_dir)
	print('mask_save_dir: ', mask_save_dir)
	print('stitch_save_dir: ', stitch_save_dir)
	
	'''
    入力、出力フォルダ辞書
	'''
	# 辞書(directories)を用意
	# パスなどを入れる[source, patch_save_dir save_dir, mask_save_dir, stitch_save_dir]
	directories = {'source': args.source, 
				   'save_dir': args.save_dir,
				   'patch_save_dir': patch_save_dir, 
				   'mask_save_dir' : mask_save_dir, 
				   'stitch_save_dir': stitch_save_dir} 


    # directories中のkeyとvalを読み込み、それぞれ表示する。
	for key, val in directories.items():
		print("{} : {}".format(key, val))
		# 出力フォルダを作成する。(入力フォルダは作らない)
		# 読み込んだkeyが'source'ではない場合、ディレクトリを作成する。
		if key not in ['source']:
			os.makedirs(val, exist_ok=True)

	'''
　　パラメータ辞書(子) デフォルト値を設定
	'''
    # セグメンテーション用のパラメータ辞書(seg_params)を用意
	# seg_level, sthresh, mthresh, close, use_otsu, keep_ids, exclude_idsをセット
	# WSI分割ダウンサンプリングレベル,セグメンテーション閾値(大->背景多),平均フィルタサイズ
    # 二値化前のモルフォロジークロージング,大津の二値化
	seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'}
	# フィルタ用のパラメータ辞書(filter_params)を用意 (組織領域の検知閾値、穴の検知閾値、穴の最大数)
	filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}
	# 表示用のパラメータ辞書を用意 (表示解像度, 線の太さ)
	vis_params = {'vis_level': -1, 'line_thickness': 250}
	# パッチ用のパラメータ辞書を用意 (パディング使用、前景か背景か判断)
	patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

	'''
	事前にパラメータを記載したcsvファイルを用意している場合
	preset_dfにデータを入れて、上記の辞書の中身を上書きする。
	'''
    # 引数presetがある場合、presetsフォルダ内のcsvファイルを読み込む
	# preset_df = presets + 引数preset
	if args.preset:
		preset_df = pd.read_csv(os.path.join('presets', args.preset))
		for key in seg_params.keys():
			seg_params[key] = preset_df.loc[0, key]

		for key in filter_params.keys():
			filter_params[key] = preset_df.loc[0, key]

		for key in vis_params.keys():
			vis_params[key] = preset_df.loc[0, key]

		for key in patch_params.keys():
			patch_params[key] = preset_df.loc[0, key]
	
	'''
    パラメータ辞書(親)
	上記の子辞書をまとめて親辞書(parameter)に入れる。
    '''	
	parameters = {'seg_params': seg_params,
				  'filter_params': filter_params,
	 			  'patch_params': patch_params,
				  'vis_params': vis_params}

    # パラメータを全表示
	print(parameters)
	'''
	実行関数の呼び出し
	(set_and_patch関数)
	'''
    # 引数は、ディレクトリ辞書、パラメータ辞書、パッチサイズ(256?)、seg,use_default_params=False、save_mask = True、
	# stitch引数、 patch_level引数、 patch引数、 process_list、オートスキップ引数, [新]ann_path=アノテーションjsonファイルパス
	seg_times, patch_times = seg_and_patch(**directories, **parameters,
											patch_size = args.patch_size, step_size=args.step_size, 
											seg = args.seg,  use_default_params=False, save_mask = True, 
											stitch= args.stitch,
											patch_level=args.patch_level, patch = args.patch,
											process_list = process_list, auto_skip=args.no_auto_skip,ann_path=args.ann_path)

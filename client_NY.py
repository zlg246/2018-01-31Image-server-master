import socket
import struct
import numpy as np
import os
import sys
import image_processing as ip
from image_processing import ImageViewer
import timeit
import features_block as block
import features_multi_cell as multi_cell
import features_cz_cell as cz_cell
import features_cz_wafer as cz_wafer
import features_multi_wafer as multi_wafer
import features_x3 as x3
import features_perc as perc
import scipy.ndimage as ndimage
from pprint import pprint
import parameters
import pandas as pd
import glob

HOST, PORT = "localhost", parameters.SERVER_PORT

depricated_metrics = ['ring_strength', 'dark_corners', 'bright_lines_mean',
                      'num_bright_lines', 'num_broken_fingers', 'bright_lines_sum',
                      'bright_area_sum', 'bright_lines_mean', 'dark_spots_area',
                      'pad_top', 'pad_bottom', 'hist_percentile_50']
depricated_images = ['ov_impure_u8']


def send_data(sock, data):
    bytes_sent = 0
    bytes_to_send = len(data)
    while bytes_sent < bytes_to_send:
        bytes_sent += sock.send(data[bytes_sent:bytes_sent + 4096])


def get_data(sock, bytes_expected):
    data_array = bytearray(bytes_expected)
    bytes_received = 0
    while bytes_received < bytes_expected:
        left = bytes_expected - bytes_received
        data = sock.recv(left)
        data_array[bytes_received:bytes_received + len(data)] = data
        bytes_received += len(data)

    return str(data_array)


def request(mode, display=False, send_path=False, return_path=False,
            skip_features=False, return_cropped=True, return_uncropped=False, return_outline=False):
    ###########
    # REQUEST #
    ###########
    param_names_float = ["verbose", "already_cropped",
                         "skip_features", "return_cropped", "return_uncropped", "return_outline",
                         "ORIGINAL_ORIENTATION"]
    param_vals_float = [0, 0,
                        int(skip_features), int(return_cropped), int(return_uncropped), int(return_outline),
                        1]
    params_dict = dict(zip(param_names_float, param_vals_float))
    param_names_str = []
    param_vals_str = []
    if return_path:
        param_names_str.append("im_output_path")
        param_vals_str.append("C:\Users\Neil\Desktop\im_out")
    images = None

    # assemble image data
    print "Mode = %d" % mode
    if mode == 0:
        msg = struct.pack('=B', mode)
        # send to server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
        send_data(sock, msg)
        response = get_data(sock, 1)
        success = struct.unpack('B', response)[0]
        print "Success: %s" % str(success == 0)
        return [], []
    if mode == 10:
        fn = r"C:\Users\Neil\BT\Data\R2 FFT\multi\raw 10 sec.tif"
    elif mode == 40:
        if int(params_dict['already_cropped']) == 0:
            fn = r"C:\Users\Neil\BT\Data\blocks\B4\693 - PL Image B4 W2 4V (PL Image - Composite).tif"
        else:
            fn = r"C:\Users\Neil\BT\Data\blocks\2015-08\tifs\120815_ISE_E_nf_14A_22C_PL_600000-dark&FFcor_cropped.tif"
    elif mode in [70, 71]:
        if mode == 70:
            fn = r"C:\Users\Neil\BT\Data\slugs\zhonghuan\tifs\219609 - 160-1-6 (Uncalibrated PL Image).tif"
        elif mode == 71:
            fn = r"C:\Users\Neil\BT\Data\slugs\pseudo round\2861 - THICK SAMPLE TEST-2 %28Uncalibrated PL Image%29.tif"
        param_names_float += ['rds_percent', 'slug_radius']
        param_vals_float += [50, 0]
    elif mode == 80:
        # PERC mono cell
        # fn = r"C:\Users\Neil\BT\Data\C3\perc\mono\BAC_1024_100\20150910_122155.612_BAC_1024_100_201.tif"
        # fn = r"C:\Users\Neil\BT\Data\cropping_test_set\cells\tifs\plg.meas.cell.plqrs.a.img.tif"
        fn = r"C:\Users\Neil\BT\Data\C3\perc\mono\BAC_1024_100\20150910_122155.612_BAC_1024_100_201.tif"
        if int(params_dict['already_cropped']) == 1:
            fn = os.path.join(r"C:\Users\Neil\BT\Data\cropped", os.path.split(fn)[1])
    elif mode == 81:
        # PERC multi cell
        fn = r"C:\Users\Neil\BT\Data\C3\perc\multi\Point\1329 - REC test E1 PL Image (PL Open-circuit Image).tif"
        if int(params_dict['already_cropped']) == 1:
            fn = os.path.join(r"C:\Users\Neil\BT\Data\cropped", os.path.split(fn)[1])
    elif mode == 82:
        # mono cell
        fn = r"C:\Users\Neil\BT\Data\C3\mono\INES_c-Si_100_1024\20150908_175300.680_INES_c-Si_100_1024_46.tif"
        if True:
            param_names_float.append("no_post_processing")
            param_vals_float.append(1)
        if int(params_dict['already_cropped']) == 1:
            fn = os.path.join(r"C:\Users\Neil\BT\Data\cropped", os.path.split(fn)[1])
    elif mode == 83:
        # multi cell
        fn = r"C:\Users\Neil\BT\Data\C3\multi\misc\20170302T110107.328_Batch 3_ID467.tif"
        # fn = r"C:\Users\Neil\BT\Data\C3\multi\Astronergy\20170831T153538.783_zt-DJ--5_ID-8.tif"
        if int(params_dict['already_cropped']) == 1:
            fn = os.path.join(r"C:\Users\Neil\BT\Data\cropped", os.path.split(fn)[1])
    elif mode == 84:
        # mono wafer
        # fn = r"C:\Users\Neil\BT\Data\CIC\cracks\tifs\S0067_20140821.131519_VI_PL21F_ID10063_GRADEB1_BIN2_raw_image.tif"
        # fn = r"C:\Users\Neil\BT\Data\mono wafer\2015-10-26\S0041_20151026.161500_longi DCA 1-2_ID2_GRADEA2_BIN4_raw.tif"
        fn = r"C:\Users\Neil\Desktop\outlines\mode84.tif"
        if int(params_dict['already_cropped']) == 1:
            fn = os.path.join(r"C:\Users\Neil\BT\Data\cropped", os.path.split(fn)[1])
    elif mode == 85:
        # multi wafer
        fn = r"C:\Users\Neil\BT\Data\overlay test set\unnormalised\tifs\S0050_20120516.193034__ID10586 - Cor.tiff"
        if int(params_dict['already_cropped']) == 1:
            fn = os.path.join(r"C:\Users\Neil\BT\Data\cropped", os.path.split(fn)[1])
    elif mode == 86:
        # X3
        fn = r"C:\Users\Neil\BT\Data\X3\mono PERC\20161024_103301.320_a_00058101.tif"
        if int(params_dict['already_cropped']) == 1:
            fn = os.path.join(r"C:\Users\Neil\BT\Data\cropped", os.path.split(fn)[1])
        param_names_float += ["num_stripes", "multi", "no_stripe_images", "ORIGINAL_ORIENTATION"]
        param_vals_float += [5, 0, 1, 1]
    elif mode == 87:
        # mono stripe
        fn = r"C:\Users\Neil\BT\Data\stripe\2017-09-07 Baccini 1 in 1\S0041_20170907.120710_Baccini 1 in 1 test_ID6_raw.tif"
    elif mode == 88:
        # multi stripe
        fn = r"C:\Users\Neil\BT\Data\stripe\2017-09-07 Baccini 1 in 1\S0041_20170907.120917_Baccini 1 in 1 test_ID7_raw.tif"
    elif mode == 89:
        # QC-C3
        #fn = r"C:\Users\Neil\BT\Data\half processed\1390 - Tet P4604 PLOC 0.2s 1Sun (Uncalibrated PL Image).tif"
        fn = r"C:\Users\Neil\Desktop\outlines\mode89.tif"
    elif mode in [90, 901]:
        # plir
        if True:
            fn1 = r"C:\Users\Neil\BT\Data\blocks\PLIR\Trina\2016-05-12\5.4V W (Uncalibrated PL Image) west short pass.tif"
            fn2 = r"C:\Users\Neil\BT\Data\blocks\PLIR\Trina\2016-05-12\5.4V W (Uncalibrated PL Image) west long pass.tif"
            fn3 = r"C:\Users\Neil\BT\Data\blocks\PLIR\Trina\2016-05-12\5.4V W (Uncalibrated PL Image) west no filter.tif"
        else:
            fn1 = r"C:\Users\Neil\Desktop\B35 files for B3\Face 1\plg.meas.block.b3bl.north.shortpass.img.tif"
            fn2 = r"C:\Users\Neil\Desktop\B35 files for B3\Face 1\plg.meas.block.b3bl.north.raw.img.tif"
            fn3 = r"C:\Users\Neil\Desktop\B35 files for B3\Face 1\plg.meas.block.b3bl.north.longpass.img.tif"
        im_sp = ip.open_image(fn1, cast_long=False).astype(np.uint16)
        im_lp = ip.open_image(fn2, cast_long=False).astype(np.uint16)
        im_pl = ip.open_image(fn3, cast_long=False).astype(np.uint16)
        if True:
            images = {'im_sp': im_sp, 'im_lp': im_lp, 'im_pl': im_pl}
        else:
            images = {'im_sp': im_sp, 'im_lp': im_lp}
        fn_xfer = r"C:\Users\Neil\BT\Data\2017-09-06 TransferFunctions.TXT"
        vals = block.load_transfer(fn_xfer)
        images['im_xfer'] = vals

        if mode == 901:
            del images['im_pl']
            mode = 90
    elif mode == 92:
        # brick markers
        fn = r"C:\Users\Neil\Desktop\20160826\1267 - Ref-C-25chiller-2 (North - Shortpass Image).tif"
    elif mode == 95:
        # resolution
        fn = r"C:\Users\Neil\BT\Data\2017-09-06 new calibration target.tif"
    elif mode == 100:
        if True:
            fn_pl = r"C:\Users\Neil\BT\Data\modules\WIN-555\WIN-555_LR0245_P93_2x2_OCPL.tif"
            fn_el = r"C:\Users\Neil\BT\Data\modules\WIN-555\WIN-555_LR0160_CV43.00_2x2_EL.tif"
        else:
            fn_pl = r"C:\Users\Neil\Desktop\Processed\CNY-098\CNY-098_G00_LR0090_P93_2x2_OCPL.tif"
            fn_el = r"C:\Users\Neil\Desktop\Processed\CNY-098\CNY-098_G00_LR0090_CC10.80_2x2_EL.tif"
        im_pl = ip.open_image(fn_pl).astype(np.uint16)
        im_el = ip.open_image(fn_el).astype(np.uint16)
        images = {'im_pl': im_pl}  # , 'im_el': im_el}
        param_names_float += ["ORIGINAL_ORIENTATION"]
        param_vals_float += [0]
    elif mode == 255:
        msg = struct.pack('B', 255)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
        send_data(sock, msg)
        return [], []
    else:
        print "Unknown mode"
        sys.exit()

    if images is None:
        # open im_pl
        im = ip.open_image(fn).astype(np.uint16)
        if False:
            im = im.T
        images = {'im_pl': im}

    if False and images['im_pl'].shape[0] > 800:
        print 'WARNING: Image resized'
        images['im_pl'] = ndimage.zoom(images['im_pl'], 0.25)

    if False:
        view = ImageViewer(images['im_pl'])
        view.show()

    # gather images
    image_names = ','.join(images.keys())
    msg = struct.pack('=BI', mode, len(image_names))
    msg += image_names
    for image_name, im in images.iteritems():
        assert image_name[:2] in ['bl', 'mk', 'im', 'ov']
        if image_name == 'im_xfer':
            bit_depth = 32
        else:
            bit_depth = 16
        binning = 1
        if send_path:
            # pass by path
            msg += struct.pack('=HHBBB', 0, 0, bit_depth, binning, len(fn))
            msg += fn
        else:
            # pass data
            msg += struct.pack('=HHBB', im.shape[1], im.shape[0], bit_depth, binning)
            msg += im.ravel().tostring()

    if False:
        param_names_float = []
        param_vals_float = []
        param_names_str = []
        param_vals_str = []

    # numerical parameter list
    param_names = ','.join(param_names_float)
    msg += struct.pack('=I', len(param_names))
    msg += param_names
    msg += np.array(param_vals_float, np.float32).tostring()

    # string input parameters
    param_names = ','.join(param_names_str)
    msg += struct.pack('=I', len(param_names))
    msg += param_names
    param_vals = ','.join(param_vals_str)
    msg += struct.pack('=I', len(param_vals))
    msg += param_vals

    t1 = timeit.default_timer()

    # send to server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    send_data(sock, msg)

    ############
    # RESPONSE #
    ############

    features = {}

    # get response code
    response = get_data(sock, 1)
    success = struct.unpack('B', response)[0]
    if success != 0:
        print("Error occurred: %d" % success)
        sys.exit()

    # get images & masks
    data = get_data(sock, 4)
    image_names_length = struct.unpack('=I', data)[0]
    if image_names_length > 0:
        image_names = get_data(sock, image_names_length).split(",")
        for im_name in image_names:
            if im_name[:3] not in ['bl_', 'mk_', 'im_', 'ov_']:
                print "ERROR: Invalid image name: %s" % im_name
                sys.exit()

            data = get_data(sock, 6)
            im_w, im_h, bit_depth, binning = struct.unpack('=hhBB', data)

            if im_w == 0 or im_h == 0:
                # read from disk
                fn_len = struct.unpack('=B', get_data(sock, 1))[0]
                fn = str(get_data(sock, fn_len))
                features[im_name] = ip.open_image(fn)
            else:
                if bit_depth == 8:
                    data = get_data(sock, 4)
                    encoding_length = struct.unpack('I', data)[0]
                    png_data = get_data(sock, encoding_length)
                    features[im_name] = ip.decode_png(png_data)
                    num_pixels = features[im_name].shape[0] * features[im_name].shape[1]
                    print "%s compression: %0.1f%%" % (im_name, (100 * encoding_length) / float(num_pixels))
                elif bit_depth == 16:
                    pixel_data = get_data(sock, im_w * im_h * 2)
                    features[im_name] = np.frombuffer(pixel_data, np.uint16).reshape(im_h, im_w)
                elif bit_depth == 32:
                    pixel_data = get_data(sock, im_w * im_h * 4)
                    features[im_name] = np.frombuffer(pixel_data, np.float32).reshape(im_h, im_w)
                else:
                    print '****', im_name
    else:
        image_names = []

    # get numerical metric
    response = get_data(sock, 4)
    string_size = struct.unpack('I', response)[0]
    if string_size > 0:
        feature_names = get_data(sock, string_size)
        feature_names = feature_names.split(',')
        num_features = len(feature_names)
        bytes_expected = num_features * 4
        feature_data = get_data(sock, bytes_expected)
        feature_data = list(np.frombuffer(feature_data, np.float32))
    else:
        feature_names = []
        feature_data = []

    # get string metrics
    string_size = struct.unpack('I', get_data(sock, 4))[0]
    if string_size > 0:
        feature_names += get_data(sock, string_size).split(',')
    string_size = struct.unpack('I', get_data(sock, 4))[0]
    if string_size > 0:
        feature_data += get_data(sock, string_size).split(',')

    metric_vals = zip(feature_names, feature_data)

    ###################
    # DISPLAY RESULTS #
    ###################
    metrics = {}
    for i in range(len(feature_names)):
        features[feature_names[i]] = feature_data[i]
        metrics[feature_names[i]] = feature_data[i]

    print "Returned images:"
    for image_name in image_names:
        print "  %s" % image_name
    print "Metrics:"
    pprint(metrics)

    t2 = timeit.default_timer()
    print('Total time: %0.03f seconds' % (t2 - t1))

    rgb = None
    view = None
    if "im_cropped_u8" in features:
        if mode == 80:
            rgb = perc.create_overlay(features)
        elif mode == 81:
            rgb = perc.create_overlay_multi(features)
        elif mode == 82:
            rgb = cz_cell.create_overlay(features)
        elif mode == 83:
            rgb = multi_cell.create_overlay(features)
        elif mode == 84:
            rgb = cz_wafer.create_overlay(features)
        elif mode == 85:
            if 'skip_features' not in params_dict or params_dict['skip_features'] != 1:
                rgb = multi_wafer.create_overlay(features)
        elif mode == 86:
            rgb = x3.create_overlay(features)

    if False:
        # save cropped version for testing
        fn_cropped = os.path.join(r"C:\Users\Neil\BT\Data\cropped",
                                  os.path.split(fn)[1])
        ip.save_image(fn_cropped, features['im_cropped_u16'])

    if display and mode != 100:
        print 'Images:'
        if 'im_pl' in images:
            print '  1: Input PL image'
            im = images['im_pl']
            view = ImageViewer(im)
        e = 2
        for feature in features.keys():
            if (feature.startswith('im_') or feature.startswith('mk_') or
                    feature.startswith('ov_') or feature.startswith('bl_')):
                print '  %d: %s' % (e, feature)
                ImageViewer(features[feature])
                e += 1
        if rgb is not None:
            print '  %d: Colour overlay' % e
            e += 1
            ImageViewer(rgb)
        if view is not None:
            view.show()

    return image_names, metric_vals


def check_metrics():
    metric_list = []
    image_list = []
    alg_metric_list = {}
    alg_image_list = {}

    for server_mode in [80, 81, 82, 83, 84, 85]:
        if server_mode in [80, 81]:
            perc = 'y'
        else:
            perc = 'n'
        if server_mode in [81, 83, 85]:
            alg = 'Multi-crystalline'
        else:
            alg = 'Mono-crystalline'
        if server_mode in [80, 81, 82, 83]:
            sample = 'Cell'
        else:
            sample = 'Wafer'
        alg_metric_list["%s,%s,%s" % (sample, alg, perc)] = []
        alg_image_list["%s,%s,%s" % (sample, alg, perc)] = []

        # send request
        image_names, feature_names, metrics = request(server_mode, display=False)

        # check results
        for image_name in image_names:
            image_list.append({'Sample': sample, 'Mode': alg, 'PERC': perc, 'Image': image_name})
            alg_image_list["%s,%s,%s" % (sample, alg, perc)].append(image_name)
        for feature_name in feature_names:
            if feature_name.startswith('break') and feature_name[5:7] != '01':
                continue
            if feature_name.startswith('defect') and feature_name[6] != '_' and feature_name[6:8] != '1_':
                continue

            if feature_name.startswith('defect1_'):
                fname = 'defectX_' + feature_name[len('defectX_'):]
            elif feature_name.startswith('break01_'):
                fname = 'breakX_' + feature_name[len('break01_'):]
            else:
                fname = feature_name  # .lower()
            alg_metric_list["%s,%s,%s" % (sample, alg, perc)].append(fname)
            metric_list.append({'Sample': sample, 'Mode': alg, 'PERC': perc,
                                'Metric': fname, 'Value': metrics[feature_name]})

    # compare metrics
    fn = r"C:\Users\Neil\Desktop\BTi - Algo metrics master list - mappings.algo.metrics.csv"
    if not os.path.isfile(fn):
        print "No comparison file found"
        return

    print '\nMETRICS\n'
    df_doc = pd.read_csv(fn)
    doc_list = {}
    for _, r in df_doc.iterrows():
        if r['Algo'] != 'features':
            continue
        # print r
        if r['PERC'] == 'y':
            perc = 'y'
        else:
            perc = 'n'
        alg = r['Mode']
        sample = r['Sample']
        key = "%s,%s,%s" % (sample, alg, perc)
        if key not in doc_list:
            doc_list[key] = []
        doc_list[key].append(r['Key'])

    for alg in alg_metric_list.keys():
        print alg
        print "  Missing from spreadsheet"
        for metric in alg_metric_list[alg]:
            if (metric.startswith('input_param') or metric.startswith('param_') or
                        metric in ['runtime'] or metric.startswith('corner_') or metric == 'filename'):
                continue
            if metric not in doc_list[alg]:
                print "    - %s" % (metric)

        print "  Extra in spreadsheet"
        for metric in doc_list[alg]:
            if (metric.startswith('input_param') or metric.startswith('param_') or
                        metric in ['runtime', 'debug']):
                continue
            if metric not in alg_metric_list[alg] and metric not in depricated_metrics:
                print "    - %s" % (metric)

    # compare images
    fn = r"C:\Users\Neil\Desktop\BTi - Algo metrics master list - Algorithm Images.csv"
    if not os.path.isfile(fn):
        print "No image file found"
        return

    print '\nIMAGES\n'
    df_doc = pd.read_csv(fn)
    doc_list = {}
    for _, r in df_doc.iterrows():
        # print r
        if r['PERC'] == 'y':
            perc = 'y'
        else:
            perc = 'n'
        alg = r['Mode']
        sample = r['Sample']
        key = "%s,%s,%s" % (sample, alg, perc)
        if key not in doc_list:
            doc_list[key] = []
        doc_list[key].append(r['Key'])

    for alg in alg_image_list.keys():
        print alg
        print "  Missing from spreadsheet"
        for im_name in alg_image_list[alg]:
            if im_name not in doc_list[alg]:
                print "    - %s" % (im_name)

        print "  Extra in spreadsheet"
        for im_name in doc_list[alg]:
            if im_name not in alg_image_list[alg] and im_name not in depricated_images:
                print "    - %s" % (im_name)


def main():
    if False:
        # for testing distribution
        for server_mode in [0, 10, 40, 70, 71, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 901, 95, 255]:
            request(server_mode, display=False)
    elif False:
        # output metrics
        fns = glob.glob("4.*_images.csv") + glob.glob("4.*_metrics.csv")
        for fn in fns:
            print "Remove: %s" % fn
            os.remove(fn)
        fn_out_im = "%s_images.csv" % parameters.ver
        with open(fn_out_im, 'a') as f:
            f.write("mode,image_name\n")
        fn_out_metrics = "%s_metrics.csv" % parameters.ver
        with open(fn_out_metrics, 'a') as f:
            f.write("mode,metric_name,sample_value\n")
        for server_mode in [0, 10, 40, 70, 71, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 95, 255]:
            image_names, metric_vals = request(server_mode, display=False)
            with open(fn_out_im, 'a') as f:
                for image_name in image_names:
                    f.write("%d,%s\n" % (server_mode, image_name))
            with open(fn_out_metrics, 'a') as f:
                for (metric_name, metric_val) in metric_vals:
                    if metric_name.startswith('input_param') or metric_name in ['runtime', 'param_alg_version']:
                        continue
                    try:
                        mv = "%0.02f" % metric_val
                    except:
                        mv = metric_val
                    f.write("%d,%s,%s\n" % (server_mode, metric_name, mv))
    elif False:
        # output metrics
        check_metrics()
    elif False:
        mode = 89
        request(mode, skip_features=False, return_cropped=False, return_uncropped=False, return_outline=False)
        request(mode, skip_features=False, return_cropped=False, return_uncropped=True, return_outline=False)
        request(mode, skip_features=False, return_cropped=True, return_uncropped=False, return_outline=False)
        request(mode, skip_features=False, return_cropped=True, return_uncropped=True, return_outline=False)
        request(mode, skip_features=True, return_cropped=False, return_uncropped=False, return_outline=False)
        request(mode, skip_features=True, return_cropped=False, return_uncropped=True, return_outline=False)
        request(mode, skip_features=True, return_cropped=True, return_uncropped=False, return_outline=False)
        request(mode, skip_features=True, return_cropped=True, return_uncropped=True, return_outline=False)
    else:
        # test one
        request(87, display=True, send_path=False, return_path=False,
                skip_features=False, return_cropped=True, return_uncropped=False, return_outline=False)


if __name__ == "__main__":
    main()

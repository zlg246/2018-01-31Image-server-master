import SocketServer
import struct
import FF
import numpy as np
import datetime
import image_processing as ip
import features_multi_wafer as multi_wafer
import features_cz_wafer as cz_wafer
import features_cz_cell as cz_cell
import features_multi_cell as multi_cell
import features_perc as perc
import features_slugs as slugs
import features_x3 as x3
import features_module as m1
import features_resolution as resolution
import features_stripes as stripe
import features_qc_c3 as qc
import cell_processing as cell
import sys
import features_block as block
import traceback
import parameters
import cropping
import timeit
import os
import cv2
from scipy import ndimage

OUTPUT_TIMING = False


def update_corner_features(features, crop_props):
    features['corner_tl_x'] = crop_props['corners'][0][1]
    features['corner_tl_y'] = crop_props['corners'][0][0]
    features['corner_tr_x'] = crop_props['corners'][1][1]
    features['corner_tr_y'] = crop_props['corners'][1][0]
    features['corner_br_x'] = crop_props['corners'][2][1]
    features['corner_br_y'] = crop_props['corners'][2][0]
    features['corner_bl_x'] = crop_props['corners'][3][1]
    features['corner_bl_y'] = crop_props['corners'][3][0]


class ThreadedServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
    pass


class ThreadedRequestHandler(SocketServer.BaseRequestHandler):
    """
    The RequestHandler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """

    def send_data(self, data):
        bytes_sent = 0
        bytes_to_send = len(data)
        while bytes_sent < bytes_to_send:
            bytes_sent += self.request.send(data[bytes_sent:bytes_sent + 8192])

    def get_data(self, bytes_expected):
        data_array = bytearray(bytes_expected)
        bytes_received = 0
        while bytes_received < bytes_expected:
            data = self.request.recv(bytes_expected - bytes_received)
            data_array[bytes_received:bytes_received + len(data)] = data
            bytes_received += len(data)

        return str(data_array)

    def handle(self):
        reload(parameters)

        # self.request is the TCP socket connected to the client
        # get the image dimensions, which is contain in the first two
        #  unsigned shorts (two bytes each)
        start_time = str(datetime.datetime.now())
        mode = struct.unpack('B', self.get_data(1))[0]
        print('Request received at %s (mode=%d)' % (start_time, mode))

        if mode == 255:
            print('  Mode: Exit')
            self.server.shutdown()
            return

        if mode == 0:
            msg = struct.pack('=B', 0)
            self.send_data(msg)
            return

        # get input images
        image_desc_length = struct.unpack('=I', self.get_data(4))[0]
        if image_desc_length == 0:
            print "ERROR: No images passed as input"
            return
        image_names_in = self.get_data(image_desc_length).split(',')
        images = {}
        for im_name in image_names_in:
            data = self.get_data(6)
            width, height, bit_depth, binning = struct.unpack('=HHBB', data)
            num_pixels = width * height
            if num_pixels == 0:
                # read from disk
                fn_len = struct.unpack('=B', self.get_data(1))[0]
                fn = str(self.get_data(fn_len))
                images[im_name] = ip.open_image(fn)
            else:
                if bit_depth == 8:
                    pixel_data = self.get_data(num_pixels)
                    im_data = np.frombuffer(pixel_data, np.uint8)
                elif bit_depth == 16:
                    pixel_data = self.get_data(num_pixels * 2)
                    im_data = np.frombuffer(pixel_data, np.uint16)
                elif bit_depth == 32:
                    pixel_data = self.get_data(num_pixels * 4)
                    im_data = np.frombuffer(pixel_data, np.float32)
                images[im_name] = im_data.reshape(height, width).astype(np.float32)

        # get numerical parameters
        data = self.get_data(4)
        param_desc_length = struct.unpack('=I', data)[0]
        if param_desc_length > 0:
            param_names = self.get_data(param_desc_length).split(",")
            num_params = len(param_names)
            param_data = self.get_data(num_params * 4)
            params_array = list(np.frombuffer(param_data, np.float32))
        else:
            param_names = []
            params_array = []

        # get string parameters
        data = self.get_data(4)
        param_desc_length = struct.unpack('=I', data)[0]
        if param_desc_length > 0:
            param_names += self.get_data(param_desc_length).split(",")
            param_vals_length = struct.unpack('=I', self.get_data(4))[0]
            params_array += self.get_data(param_vals_length).split(",")

        # override defaults in parameters.py
        for pn, pv in zip(param_names, params_array):
            if pn.upper() in dir(parameters):
                setattr(parameters, pn.upper(), pv)

        # store input parameters in the features dict
        param_names = ['input_param_' + pn for pn in param_names]
        features = dict(zip(param_names, params_array))
        if 'input_param_already_cropped' in features and int(features['input_param_already_cropped']) == 1:
            already_cropped = True
        else:
            already_cropped = False

        if 'input_param_return_uncropped' in features and int(features['input_param_return_uncropped']) == 1:
            return_uncropped = True
        else:
            return_uncropped = False

        if 'input_param_return_cropped' in features and int(features['input_param_return_cropped']) == 0:
            return_cropped = False
        else:
            return_cropped = True

        if 'input_param_return_outline' in features and int(features['input_param_return_outline']) == 1:
            return_outline = True
        else:
            return_outline = False

        # call image processing algorithm
        try:
            if mode == 10:
                print('  Mode: Hash Pattern correction')
                im_raw = images['im_pl'].astype(np.float32)
                im_corrected = FF.correct_hash_pattern(im_raw)
                features['im_corrected_u16'] = im_corrected.astype(np.uint16)
            elif mode == 40:
                print('  Mode: Block processing')
                im = images['im_pl'].astype(np.float32)
                block.feature_extraction(im, features, crop=not already_cropped)
                features['crop_left'] = features['_crop_bounds'][0]
                features['crop_right'] = features['_crop_bounds'][1]
                features['crop_top'] = features['_crop_bounds'][2]
                features['crop_bottom'] = features['_crop_bounds'][3]
                features['bl_cropped_u8'] = np.zeros_like(features['im_cropped_u8'], np.uint8)

                if return_uncropped or return_outline:
                    left, right, top, bottom = features['_crop_bounds']
                    mask = np.ones_like(images['im_pl'], np.uint8)
                    mask[top:bottom, left:right] = 0
                    if abs(features['crop_rotation']) > 0.01:
                        h, w = mask.shape
                        rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), features['crop_rotation'] * -1, 1.0)
                        mask = cv2.warpAffine(mask, rot_mat, (w, h), flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_REPLICATE)  # .astype(np.uint8)
                    if return_uncropped:
                        features['bl_uncropped_u8'] = mask
            elif mode in [70, 71]:
                print('  Mode: Slugs')
                im = images['im_pl'].astype(np.float32)
                if 'input_param_rds_percent' not in features:
                    features['param_rds_percent'] = 50
                else:
                    features['param_rds_percent'] = int(features['input_param_rds_percent'])
                if 'param_radius_prior' not in features:
                    features['param_radius_prior'] = 0
                else:
                    features['param_radius_prior'] = int(features['input_param_slug_radius'])
                slugs.feature_extraction(im, features)
                update_corner_features(features, features)
                features['im_cropped_u8'] = (ip.scale_image(images['im_pl']) * 255).astype(np.uint8)
                features['im_cropped_u16'] = images['im_pl'].astype(np.uint16)
                mask = features['bl_uncropped_u8']
                if not return_uncropped:
                    del features['bl_uncropped_u8']
            elif mode in [84, 85, 89]:
                if mode == 84:
                    print('  Mode: Mono wafer')
                    im = images['im_pl'].astype(np.float32)
                    features['_alg_mode'] = 'mono wafer'
                    crop_props = cropping.crop_wafer_cz(im, create_mask=True, skip_crop=already_cropped)
                    features['corners'] = crop_props['corners']
                    features['_wafer_middle_orig'] = crop_props['center']
                    cropped = cropping.correct_rotation(im, crop_props, pad=False, border_erode=parameters.BORDER_ERODE_CZ,
                                                        fix_chamfer=False)
                    cz_wafer.feature_extraction(cropped, crop_props, features=features)
                    update_corner_features(features, crop_props)
                elif mode == 85:
                    print('  Mode: Multi wafer')
                    im = images['im_pl'].astype(np.float32)
                    features['_alg_mode'] = 'multi wafer'
                    if not already_cropped:
                        crop_props = cropping.crop_wafer(im, create_mask=True)
                        features['corners'] = crop_props['corners']
                        cropped = cropping.correct_rotation(im, crop_props, pad=False, border_erode=parameters.BORDER_ERODE)
                    else:
                        crop_props = {}
                        crop_props['estimated_width'] = im.shape[0]
                        crop_props['center'] = (im.shape[0] / 2, im.shape[1] / 2)
                        crop_props['corners'] = [[0, 0],
                                                 [0, im.shape[1]],
                                                 [im.shape[0], im.shape[1]],
                                                 [im.shape[0], 0],
                                                 ]
                        crop_props['corners_floats'] = crop_props['corners']
                        crop_props['estimated_rotation'] = 0
                        crop_props['mask'] = np.ones_like(im, np.uint8)
                        cropped = im
                    multi_wafer.feature_extraction(cropped, crop_props, features=features)
                    multi_wafer.combined_features(features)
                    update_corner_features(features, crop_props)
                elif mode == 89:
                    print('  Mode: QC-C3')
                    features['_alg_mode'] = 'qc'
                    im = images['im_pl'].astype(np.float32)
                    crop_props = qc.feature_extraction(im, features)

                if return_uncropped:
                    features['bl_uncropped_u8'] = crop_props['mask']
            elif mode in [80, 81, 82, 83, 86, 87, 88]:
                if mode == 80:
                    print('  Mode: PERC mono')
                    im = images['im_pl'].astype(np.float32)
                    features['_alg_mode'] = 'perc mono'
                    perc.feature_extraction(im, features, already_cropped=already_cropped)
                elif mode == 81:
                    print('  Mode: PERC multi')
                    im = images['im_pl'].astype(np.float32)
                    features['_alg_mode'] = 'perc multi'
                    perc.feature_extraction_multi(im, features, already_cropped=already_cropped)
                elif mode == 82:
                    print('  Mode: Mono cells')
                    im = images['im_pl'].astype(np.float32)
                    features['_alg_mode'] = 'mono cell'
                    cz_cell.feature_extraction(im, features, skip_crop=already_cropped)
                elif mode == 83:
                    print('  Mode: Multi cells')
                    im = images['im_pl'].astype(np.float32)
                    features['_alg_mode'] = 'multi cell'
                    multi_cell.feature_extraction(im, features, already_cropped=already_cropped)
                elif mode == 86:
                    print('  Mode: X3')
                    features['_alg_mode'] = 'x3'
                    im = images['im_pl'].astype(np.float32)
                    x3.feature_extraction(im, features, already_cropped=already_cropped)
                elif mode == 87:
                    print('  Mode: Stripe (mono)')
                    features['_alg_mode'] = 'stripe'
                    features['_cell_type'] = 'mono'
                    im = images['im_pl'].astype(np.float32)
                    stripe.feature_extraction(im, features, skip_crop=already_cropped)
                elif mode == 88:
                    print('  Mode: Stripe (multi)')
                    features['_alg_mode'] = 'stripe'
                    features['_cell_type'] = 'multi'
                    im = images['im_pl'].astype(np.float32)
                    stripe.feature_extraction(im, features, skip_crop=already_cropped)
                update_corner_features(features, features)

                if return_uncropped:
                    mask = features['bl_cropped_u8']
                    im_h, im_w = im.shape
                    if 'cell_rotated' in features and features['cell_rotated']:
                        if parameters.ORIGINAL_ORIENTATION:
                            mask = mask[:, ::-1].T
                        im_h = im.shape[1]
                        im_w = im.shape[0]

                    # undo rotation and cropping
                    mask = np.pad(mask, ((features['crop_top'], im_h - features['crop_bottom']),
                                         (features['crop_left'], im_w - features['crop_right'])),
                                  mode='constant', constant_values=((1, 1), (1, 1)))

                    # created rotated version of full image
                    mask_rotated = np.empty(im.shape, np.float32)
                    h, w = mask.shape
                    if 'cell_rotated' not in features or not features['cell_rotated']:
                        rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), -features['crop_rotation'], 1.0)
                    else:
                        rot_mat = cv2.getRotationMatrix2D((h // 2, h // 2), -features['crop_rotation'], 1.0)
                    cv2.warpAffine(mask.astype(np.float32), rot_mat, (im.shape[1], im.shape[0]),
                                   flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, dst=mask_rotated, borderValue=1)
                    #print mask.shape, im.shape
                    assert mask_rotated.shape == im.shape
                    features['bl_uncropped_u8'] = np.round(mask_rotated).astype(np.uint8)
            elif mode == 90:
                print('  Mode: plir')
                im_sp = images['im_sp'].astype(np.float32)
                im_lp = images['im_lp'].astype(np.float32)
                if 'im_xfer' not in images:
                    print "ERROR: Transfer functions not found"
                    self.send_data(struct.pack('=B', 6))
                    return

                spline_plir, spline_nf, spline_sp, spline_lp = block.interpolate_transfer(images['im_xfer'])

                if 'im_pl' in images:
                    im_pl = images['im_pl'].astype(np.float32)
                    plc_found = block.plir(im_sp, im_lp, im_pl, features, spline_plir, spline_nf)
                else:
                    plc_found = block.plir2(im_sp, im_lp, features, spline_plir, spline_sp)
                if not plc_found:
                    self.send_data(struct.pack('=B', 5))
                    return

                if return_uncropped or return_outline:
                    left, right, top, bottom = features['_crop_bounds']
                    if 'im_pl' in images:
                        left *= 2
                        right *= 2
                        top *= 2
                        bottom *= 2
                        mask = np.ones_like(images['im_pl'], np.uint8)
                    else:
                        mask = np.ones_like(images['im_sp'], np.uint8)

                    mask[top:bottom, left:right] = 0
                    if abs(features['crop_rotation']) > 0.01:
                        h, w = mask.shape
                        rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), features['crop_rotation'] * -1, 1.0)
                        mask = cv2.warpAffine(mask, rot_mat, (w, h), flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_REPLICATE)  # .astype(np.uint8)
                    if return_uncropped:
                        features['bl_uncropped_u8'] = mask
            elif mode == 92:
                print('  Mode: Distance between brick markers')
                im = images['im_pl'].astype(np.float32)
                block.MarkerLineDist(im, features)
            elif mode == 95:
                print('  Mode: Pixels per mm')
                im = images['im_pl'].astype(np.float32)
                resolution.resolution(im, features)
            elif mode == 100:
                print('  Mode: M1')
                if 'im_el' in images:
                    im_el = images['im_el'].astype(np.float32)
                else:
                    im_el = None
                im_pl = images['im_pl'].astype(np.float32)
                m1.feature_extraction(im_pl, im_el, features)
            else:
                print("ERROR: Mode %d not supported" % mode)
                self.send_data(struct.pack('=B', 1))
                return

            if not return_cropped:
                for im_name in ['im_cropped_u16', 'im_cropped_u8', 'bl_cropped_u8', "im_cropped_sp_u8",
                                'im_cropped_nf_u8', 'im_cropped_sp_u16',
                                'im_cropped_nf_u16', 'im_cropped_lp_u16']:
                    if im_name in features:
                        del features[im_name]

            if return_outline:
                if mode in [40, 70, 90]:
                    binary_struct = ndimage.generate_binary_structure(2, 1)
                    foreground = 1 - mask
                    outline = foreground - ndimage.binary_erosion(foreground, binary_struct)
                    features['bl_crop_outline_u8'] = outline.astype(np.uint8)
                else:
                    features['bl_crop_outline_u8'] = cropping.draw_crop_box(im, features, mode="mask")

        except cropping.WaferMissingException:
            self.send_data(struct.pack('=B', 2))
            return
        except cell.MissingBusbarsException:
            self.send_data(struct.pack('=B', 3))
            return
        except cell.CellFingersException:
            self.send_data(struct.pack('=B', 4))
            return
        except:
            traceback.print_exc(file=sys.stdout)
            self.send_data(struct.pack('=B', 1))
            return

        # success
        msg = struct.pack('=B', 0)
        self.send_data(msg)

        # return images
        image_names = []
        for f in features.keys():
            if f.split('_')[-1] not in ['u8', 'u16', 'f32'] or f[0] == '_':
                continue
            if f[:3] not in ['bl_', 'mk_', 'im_', 'ov_']:
                print "ERROR: invalid image name: %s" % f

            image_names.append(f)
        image_names.sort()

        image_names_send = ','.join(image_names)

        self.send_data(struct.pack('I', len(image_names_send)))
        self.send_data(image_names_send)
        for im_name in image_names:
            fields = im_name.split('_')
            if fields[-1] == "u8":
                bit_depth = 8
            elif fields[-1] == "u16":
                bit_depth = 16
            elif fields[-1] == "f32":
                bit_depth = 32

            # convert binary masks from 0,1 to 0,255
            if fields[0] == 'mk' and bit_depth == 8:
                features[im_name] *= 255

            if ('input_param_im_output_path' in features and
                    len(features['input_param_im_output_path']) > 0 and bit_depth in [8, 16]):
                # send back as path.
                msg = struct.pack('=hhBB', 0, 0, 0, 1)
                if bit_depth == 8:
                    ext = '.png'
                else:
                    ext = '.tif'
                fn_out = os.path.join(features['input_param_im_output_path'], im_name + ext)
                ip.save_image(fn_out, features[im_name], scale=False)
                fn_len = len(fn_out)
                msg += struct.pack('=B', fn_len)
                msg += fn_out
            else:
                # image data
                height, width = features[im_name].shape
                binning = 1
                msg = struct.pack('=hhBB', width, height, bit_depth, binning)

                if fields[-1] == "u8":
                    png = ip.encode_png(features[im_name])
                    msg += struct.pack('=I', len(png))
                    msg += png
                elif fields[-1] in ["u16", "f32"]:
                    msg += features[im_name].tostring()

            self.send_data(msg)

        # numerical features
        feature_names = []
        feature_vals = []
        for k in features.keys():
            if (k in ['cropped', 'corners', 'filename', 'center'] or
                    k.startswith("bl_") or k.startswith('_') or k.startswith("mask_") or
                    k.startswith("mk_") or k.startswith("im_") or k.startswith("ov_")):
                continue
            if type(features[k]) is str:
                continue
            feature_names.append(k)
        feature_names.sort()
        for feature in feature_names:
            feature_vals.append(float(features[feature]))
        feature_names = ','.join(feature_names)
        feature_vals = np.array(feature_vals, np.float32)
        bytes_to_send = len(feature_names)
        self.send_data(struct.pack('=I', bytes_to_send))
        self.send_data(feature_names)
        msg = feature_vals.ravel().tostring()
        self.send_data(msg)

        # string features
        feature_names = []
        feature_vals = []
        for k in features.keys():
            if k.startswith('_'):
                continue
            if type(features[k]) is not str:
                continue
            feature_names.append(k)
        feature_names.sort()
        for feature in feature_names:
            feature_vals.append(features[feature])
        feature_names = ','.join(feature_names)
        feature_vals = ','.join(feature_vals)
        bytes_to_send = len(feature_names)
        self.send_data(struct.pack('=I', bytes_to_send))
        if bytes_to_send > 0:
            self.send_data(feature_names)
        bytes_to_send = len(feature_vals)
        self.send_data(struct.pack('=I', bytes_to_send))
        if bytes_to_send > 0:
            self.send_data(feature_vals)

        return


if __name__ == "__main__":
    # Create the server, binding to HOST on PORT
    server = ThreadedServer((parameters.HOST, parameters.SERVER_PORT), ThreadedRequestHandler)

    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    print("Algorithm Server started at %s" % (datetime.datetime.now()))
    print("Waiting for requests on %s, PORT: %d..." % (parameters.HOST, parameters.SERVER_PORT))

    server.serve_forever()
    print "Goodbye."

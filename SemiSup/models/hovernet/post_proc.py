import os

import cv2
import numpy as np

from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import (
    binary_dilation,
    binary_fill_holes,
    distance_transform_cdt,
    distance_transform_edt,
)

from skimage.segmentation import watershed
from scipy import ndimage

import warnings


def noop(*args, **kargs):
    pass


warnings.warn = noop

####
def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def remove_small_objects(pred, min_size=64, connectivity=1):
    """Remove connected components smaller than the specified size.

    This function is taken from skimage.morphology.remove_small_objects, but the warning
    is removed when a single label is provided.

    Args:
        pred: input labelled array
        min_size: minimum size of instance in output array
        connectivity: The connectivity defining the neighborhood of a pixel.

    Returns:
        out: output array with instances removed under min_size

    """
    out = pred

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndimage.generate_binary_structure(pred.ndim, connectivity)
        ccs = np.zeros_like(pred, dtype=np.int32)
        ndimage.label(pred, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out


####
def __proc_np_hv(pred, image_name):
    """Process Nuclei Prediction with XY Coordinate Map.

    Args:
        pred: prediction output, assuming 
              channel 0 contain probability map of nuclei
              channel 1 containing the regressed X-map
              channel 2 containing the regressed Y-map

    """
    pred = np.array(pred, dtype=np.float32)

    blb_raw = pred[..., 0]
    h_dir_raw = pred[..., 1]
    v_dir_raw = pred[..., 2]

    # processing
    blb = np.array(blb_raw >= 0.5, dtype=np.int32)

    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1  # background is 0 already

    # cv2.imwrite(f'/mnt/Disk1/my/NuBaseline/baseline3/proc_check/{image_name}/{image_name}_np.jpg', blb)

    # cv2.imwrite(f'/mnt/Disk1/my/NuBaseline/baseline3/proc_check/{image_name}/{image_name}_hmap.jpg', h_dir_raw)
    # cv2.imwrite(f'/mnt/Disk1/my/NuBaseline/baseline3/proc_check/{image_name}/{image_name}_vmap.jpg', v_dir_raw)

    h_dir = cv2.normalize(
        h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    v_dir = cv2.normalize(
        v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    # cv2.imwrite(f'/mnt/Disk1/my/NuBaseline/baseline3/proc_check/{image_name}/{image_name}_hmap_norm.jpg', h_dir * 255)
    # cv2.imwrite(f'/mnt/Disk1/my/NuBaseline/baseline3/proc_check/{image_name}/{image_name}_vmap_norm.jpg', v_dir * 255)

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

    sobelh = 1 - (
        cv2.normalize(
            sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )
    sobelv = 1 - (
        cv2.normalize(
            sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    ## nuclei values form mountains so inverse to get basins
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall = np.array(overall >= 0.4, dtype=np.int32)
    # cv2.imwrite(f'/mnt/Disk1/my/NuBaseline/baseline3/proc_check/{image_name}/{image_name}_overall.jpg', overall * 255)


    # cv2.imwrite(f'/mnt/Disk1/my/NuBaseline/baseline3/proc_check/{image_name}/{image_name}_res.jpg', res_show)

    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    raw_marker = np.copy(marker)
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=10)
    proced_pred = watershed(dist, markers=marker, mask=blb)

    ################################ draw mid res ###############################
    # '''
    # refine the segmentation on patch edge
    # '''
    # inst_list = list(np.unique(proced_pred))
    # inst_list.remove(0)  # 0 is background
    # for inst_id in inst_list:
    #
    #     inst_map = np.array(proced_pred == inst_id, np.uint8)
    #
    #     inst_box = get_bounding_box(inst_map)
    #     inst_map_remove_msk = np.zeros_like(inst_map)
    #
    #     cell_size = 256
    #     (w, h) = inst_map.shape
    #     w_line = w // cell_size
    #     h_line = h // cell_size
    #     for i in range(1, w_line + 1):
    #         if inst_box[0] == 256 * i:
    #             msk = np.zeros_like(inst_map)
    #             msk[256 * i:, :] = 1
    #             inst_map_remove_msk = inst_map * msk
    #     for j in range(1, h_line + 1):
    #         if inst_box[2] == 256 * j:
    #             msk = np.zeros_like(inst_map)
    #             msk[:, 256 * j:] = 1
    #             inst_map_remove_msk = inst_map * msk
    #     # filter the markers on edge
    #     raw_marker[inst_map_remove_msk != 0] = 0
    #
    # # draw imd results
    blank = np.ones(shape=(blb.shape[0], 3)) * 255
    res_show = np.concatenate([blb * 255, blank,
                               h_dir * 255, blank,
                               v_dir * 255, blank,
                               overall * 255, blank,
                               raw_marker * 255], axis=1)

    # raw_marker = measurements.label(raw_marker)[0]
    # raw_marker = remove_small_objects(raw_marker, min_size=10)
    # proced_pred = watershed(dist, markers=raw_marker, mask=blb)

    return proced_pred, res_show


####
def process(pred_map, nr_types=None, return_centroids=False, image_name=''):
    """Post processing script for image tiles.

    Args:
        pred_map: commbined output of tp, np and hv branches, in the same order
        nr_types: number of types considered at output of nc branch
        overlaid_img: img to overlay the predicted instances upon, `None` means no
        type_colour (dict) : `None` to use random, else overlay instances of a type to colour in the dict
        output_dtype: data type of output
    
    Returns:
        pred_inst:     pixel-wise nuclear instance segmentation prediction
        pred_type_out: pixel-wise nuclear type prediction 

    """
    if nr_types is not None:
        pred_type = pred_map[..., :1]
        pred_inst = pred_map[..., 1:]
        pred_type = pred_type.astype(np.int32)
    else:
        pred_inst = pred_map

    pred_inst = np.squeeze(pred_inst)

    pred_inst, res_show = __proc_np_hv(pred_inst, image_name)

    inst_info_dict = None
    if return_centroids or nr_types is not None:
        inst_id_list = np.unique(pred_inst)[1:]  # exlcude background
        inst_info_dict = {}
        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id
            # TODO: chane format of bbox output
            rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
            inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
            inst_map = inst_map[
                inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]
            ]
            inst_map = inst_map.astype(np.uint8)
            inst_moment = cv2.moments(inst_map)
            inst_contour = cv2.findContours(
                inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # * opencv protocol format may break
            inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
            # < 3 points dont make a contour, so skip, likely artifact too
            # as the contours obtained via approximation => too small or sthg
            if inst_contour.shape[0] < 3:
                continue
            if len(inst_contour.shape) != 2:
                continue # ! check for trickery shape
            inst_centroid = [
                (inst_moment["m10"] / inst_moment["m00"]),
                (inst_moment["m01"] / inst_moment["m00"]),
            ]
            inst_centroid = np.array(inst_centroid)
            inst_contour[:, 0] += inst_bbox[0][1]  # X
            inst_contour[:, 1] += inst_bbox[0][0]  # Y
            inst_centroid[0] += inst_bbox[0][1]  # X
            inst_centroid[1] += inst_bbox[0][0]  # Y
            inst_info_dict[inst_id] = {  # inst_id should start at 1
                "bbox": inst_bbox,
                "centroid": inst_centroid,
                "contour": inst_contour,
                "type_prob": None,
                "type": None,
            }

    if nr_types is not None:
        #### * Get class of each instance id, stored at index id-1
        for inst_id in list(inst_info_dict.keys()):
            rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["bbox"]).flatten()
            inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
            inst_type_crop = pred_type[rmin:rmax, cmin:cmax]
            inst_map_crop = (
                inst_map_crop == inst_id
            )  # TODO: duplicated operation, may be expensive
            inst_type = inst_type_crop[inst_map_crop]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0:  # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
            type_dict = {v[0]: v[1] for v in type_list}
            type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
            inst_info_dict[inst_id]["type"] = int(inst_type)
            inst_info_dict[inst_id]["type_prob"] = float(type_prob)

    # print('here')
    # ! WARNING: ID MAY NOT BE CONTIGUOUS
    # inst_id in the dict maps to the same value in the `pred_inst`
    return pred_inst, inst_info_dict, res_show

import pvl
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import re
from skimage.restoration import richardson_lucy
from scipy.optimize import curve_fit, least_squares
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.interpolate import UnivariateSpline
from numpy.polynomial import Polynomial
from osgeo import gdal
import pywt
from matplotlib.colors import Normalize, to_rgba
planck_constant = 6.62607015e-34
speed_of_light = 2.99792458e8
boltzmann_constant = 1.380649e-23 

FILTER_BANDS_VIS = {221, 222}
FILTER_BANDS_IR = (
    list(range(44, 59)) +
    list(range(148, 169)) +
    list(range(288, 298)) +
    list(range(353, 373))
)
defective_vis_pixels = [
    (30, 308), (31, 308), (47, 409), (48, 187), (48, 188), (49, 59), (54, 137), (71, 215),
    (100, 78), (108, 413), (109, 19), (111, 19), (114, 424), (118, 363), (126, 410), (130, 292),
    (136, 271), (139, 235), (147, 222), (150, 54), (150, 59), (150, 78), (160, 372),
    (162, 36), (162, 37), (162, 248), (162, 330), (163, 36), (163, 37), (163, 248), (163, 330),
    (165, 32), (166, 32), (166, 173), (168, 232), (169, 363), (172, 189), (173, 92),
    (175, 228), (175, 266), (175, 267), (176, 152), (176, 229), (177, 155), (179, 196),
    (181, 249), (183, 354), (186, 238), (186, 387), (188, 276), (188, 352), (189, 294),
    (189, 352), (189, 391), (189, 413), (190, 195), (191, 411), (194, 358), (196, 266),
    (196, 362), (199, 23), (199, 24), (203, 257), (203, 370), (204, 257), (207, 265), (211, 291),
    (216, 287), (222, 249), (222, 338), (223, 339), (223, 340), (225, 274), (227, 103), (229, 248),
    (234, 306), (234, 424), (238, 249), (238, 277), (238, 416), (238, 417), (239, 405), (241, 15), 
    (241, 16), (241, 386), (241, 387), (242, 15), (242, 16), (242, 364), (245, 128), (248, 304),
    (248, 305), (250, 223), (251, 223), (252, 274), (253, 307)
]
defective_ir_pixels = [
    (8, 86),(12, 148),(16, 327),(20, 39), (20, 40), (20, 41), (20, 42), (20, 43),
    (21, 39), (21, 42),(22, 40), (22, 42),(24, 37),(35, 218),(45, 337),(46, 212),
    (52, 280),(59, 430),(74, 121),(79, 186),(82, 190),(83, 130),(86, 182),(88, 200),
    (90, 203),(94, 189),(99, 73),(100, 73),(101, 72), (101, 223), (101, 224),
    (102, 223), (102, 225),(103, 223),(111, 304),(112, 28),(121, 193),(128, 149), (128, 172),
    (129, 172),(130, 159), (130, 187),(132, 132),(138, 383), (138, 384),(140, 202),
    (142, 344),(143, 343),(144, 343),(145, 343),(146, 342),(147, 344),(148, 108),
    (149, 169), (149, 170),(155, 1),(156, 1), (156, 9), (156, 198),
    (157, 1), (157, 15), (157, 25),(158, 9), (158, 17),(159, 14), (159, 18),
    (160, 19), (160, 28), (160, 29),(161, 26), (161, 28), (161, 29), (161, 181),(162, 181),
    (171, 57), (171, 58), (171, 59), (171, 60), (171, 61), (171, 62), (171, 63), (171, 64),
    (172, 57), (172, 58), (172, 59), (172, 60), (172, 61), (172, 62), (172, 63), (172, 64), (172, 227),
    (173, 59), (173, 60), (173, 61), (173, 62), (173, 63), (173, 64), (173, 65), (173, 66), (173, 67), (173, 68),
    (174, 60), (174, 61), (174, 62), (174, 63), (174, 64), (174, 65), (174, 66), (174, 67),
    (175, 61), (175, 62), (175, 63),
    (191, 111), (191, 112),(192, 110), (192, 111), (192, 112), (192, 113),
    (193, 111), (193, 112), (193, 245), (193, 246), (193, 328), (193, 333), (193, 384),
    (219, 428),(227, 211),(228, 79), (228, 222),(229, 116),(234, 210),
    (235, 175), (235, 226),(236, 186),(237, 129),(238, 38),(241, 233),
    (243, 202),(244, 228),(245, 191), (245, 192),(250, 414)
]

# def load_lbl_ignoring_comments(lbl_path):
#     with open(lbl_path, 'r', encoding='latin-1') as f:
#         lines = f.readlines()

#     valid_lines = []
#     for line in lines:
#         line = line.strip()
#         if line.startswith("/*"):
#             continue
#         if line.startswith("END"):
#             break
#         # skip obvious garbage lines (like B / or malformed lines)
#         if not line or '"' in line and "/" in line:
#             continue
#         valid_lines.append(line)

#     try:
#         return pvl.loads("\n".join(valid_lines))
#     except Exception as e:
#         print(f"Failed to parse {lbl_path}: {e}")
#         raise


def parse_lbl_metadata(lbl_path):
    lbl_data = pvl.load(lbl_path)
    qube = lbl_data.get("QUBE", {})
    bands, samples, lines = qube.get("CORE_ITEMS", [None]*3)
    item_bytes = qube.get("CORE_ITEM_BYTES")
    item_type = qube.get("CORE_ITEM_TYPE")
    dtype_map = {
        (2, "MSB_INTEGER"): ">i2",
        (2, "LSB_INTEGER"): "<i2",
        (4, "MSB_INTEGER"): ">i4",
        (4, "LSB_INTEGER"): "<i4",
        (4, "IEEE_REAL"): ">f4",
        (8, "IEEE_REAL"): ">f8",
    }
    dtype = dtype_map.get((item_bytes, item_type))
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {item_bytes} {item_type}")

    band_bin = qube.get("BAND_BIN", {})

    def parse_time(key):
        val = lbl_data.get(key)
        if isinstance(val, str):
            try:
                return datetime.strptime(val, "%Y-%m-%dT%H:%M:%S.%f")
            except ValueError:
                return datetime.strptime(val, "%Y-%m-%dT%H:%M:%S")
        return val

    def get_float(key, default=np.nan):
        try:
            return float(lbl_data.get(key, default))
        except Exception:
            return default

    def get_str(key, default=""):
        val = lbl_data.get(key)
        return str(val).strip() if val else default
    exposure_time = None
    external_repetition_time = None
    if "FRAME_PARAMETER" in lbl_data and "FRAME_PARAMETER_DESC" in lbl_data:
        try:
            param_desc = lbl_data["FRAME_PARAMETER_DESC"]
            param_values = lbl_data["FRAME_PARAMETER"]
            idx_exp = param_desc.index("EXPOSURE_DURATION")
            exposure_time = float(param_values[idx_exp])
            idx_ert = param_desc.index("EXTERNAL_REPETITION_TIME")
            external_repetition_time = float(param_values[idx_ert])
        except Exception:
            pass

    return {
        "shape": (bands, samples, lines),
        "dtype": dtype,
        "core_null": qube.get("CORE_NULL", -32768),
        "core_low_saturation": qube.get("CORE_LOW_REPR_SATURATION", -32767),
        "core_high_saturation": qube.get("CORE_HIGH_REPR_SATURATION", -32767),
        "core_multiplier": qube.get("CORE_MULTIPLIER", 1.0),
        "core_base": qube.get("CORE_BASE", 0.0),
        "product_type": lbl_data.get("PRODUCT_TYPE", "UNKNOWN"),
        "wave_length_cen": np.array(band_bin.get("BAND_BIN_CENTER", []), dtype=np.float32),
        "wave_width": np.array(band_bin.get("BAND_BIN_WIDTH", []), dtype=np.float32),
        "wave_length_band_val": np.array(band_bin.get("BAND_BIN_ORIGINAL_BAND", []), dtype=np.int16),
        "spacecraft_solar_dist": float(qube.get("SPACECRAFT_SOLAR_DISTANCE", 441765159.0)),
        "start_time": parse_time("START_TIME"),
        "end_time": parse_time("STOP_TIME"),
        "exposure_time": exposure_time,
        "external_repetition_time": external_repetition_time,
        "mission_phase": get_str("MISSION_PHASE_NAME"),
        "solar_incidence": get_float("INCIDENCE_ANGLE"),
        "emission_angle": get_float("EMISSION_ANGLE"),
        "phase_angle": get_float("PHASE_ANGLE"),
        "sc_target_distance": get_float("TARGET_CENTER_DISTANCE"),
        "sub_spacecraft_lat": get_float("SUB_SPACECRAFT_LATITUDE"),
        "sub_spacecraft_lon": get_float("SUB_SPACECRAFT_LONGITUDE"),
        "local_hour_angle": get_float("LOCAL_HOUR_ANGLE"),
        "target_name": get_str("TARGET_NAME"),
        "target_type": get_str("TARGET_TYPE"),
    }

def fix_endianness(array, dtype):
    dtype = np.dtype(dtype)
    if dtype.byteorder not in ('=', '|') and dtype.byteorder != sys.byteorder:
        array = array.byteswap().view(dtype.newbyteorder())
    return array
def load_qub_from_lbl(lbl_path, qub_path, cal=False, return_flag_mask=False):
    meta = parse_lbl_metadata(lbl_path)
    shape = meta["shape"]
    dtype = meta["dtype"]
    with open(qub_path, "rb") as f:
        data = np.fromfile(f, dtype=dtype)
    data = fix_endianness(data, dtype).reshape(shape, order="F").astype(np.float32)
    null, low, high = meta["core_null"], meta["core_low_saturation"], meta["core_high_saturation"]
    flag_mask = np.zeros_like(data, dtype=np.uint8)  # 0 = valid
    flag_mask[data == null] = 1  # null
    flag_mask[data == low] = 2  # low saturation
    flag_mask[data == high] = 3  # high saturation
    flag_mask[data == -32766] = 4
    flag_mask[data == -32765] = 5
    flag_mask[data == -32764] = 6
    flag_mask[data == -32762] = 7
    flag_mask[data == -32761] = 8
    data = data.astype(np.float32)
    data[flag_mask != 0] = np.nan
    if not cal:
        mult, base = meta["core_multiplier"], meta["core_base"]
        if mult != 1.0 or base != 0.0:
            data *= mult
            data += base
    if return_flag_mask:
        return data, meta, flag_mask
    else:
        return data, meta

def load_qub_from_lbl_name(base_folder, lbl_filename, cal=False, return_flag_mask=False):
    lbl_path = os.path.join(base_folder, lbl_filename)
    lbl_data = pvl.load(lbl_path)
    qube = lbl_data.get("QUBE", {})
    bands, samples, lines = qube.get("CORE_ITEMS", [None]*3)
    item_bytes = qube.get("CORE_ITEM_BYTES")
    item_type = qube.get("CORE_ITEM_TYPE")

    dtype_map = {
        (2, "MSB_INTEGER"): ">i2",
        (2, "LSB_INTEGER"): "<i2",
        (4, "MSB_INTEGER"): ">i4",
        (4, "LSB_INTEGER"): "<i4",
        (4, "IEEE_REAL"): ">f4",
        (8, "IEEE_REAL"): ">f8",
    }
    dtype = dtype_map.get((item_bytes, item_type))
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {item_bytes} {item_type}")

    def fix_endianness(array, dtype):
        dtype = np.dtype(dtype)
        if dtype.byteorder not in ('=', '|') and dtype.byteorder != sys.byteorder:
            array = array.byteswap().view(dtype.newbyteorder())
        return array

    def parse_time(key):
        val = lbl_data.get(key)
        if isinstance(val, str):
            try:
                return datetime.strptime(val, "%Y-%m-%dT%H:%M:%S.%f")
            except ValueError:
                return datetime.strptime(val, "%Y-%m-%dT%H:%M:%S")
        return val

    def get_float(key, default=np.nan):
        try:
            return float(lbl_data.get(key, default))
        except Exception:
            return default

    def get_str(key, default=""):
        val = lbl_data.get(key)
        return str(val).strip() if val else default

    band_bin = qube.get("BAND_BIN", {})
    exposure_time = None
    external_repetition_time = None
    if "FRAME_PARAMETER" in lbl_data and "FRAME_PARAMETER_DESC" in lbl_data:
        try:
            param_desc = lbl_data["FRAME_PARAMETER_DESC"]
            param_values = lbl_data["FRAME_PARAMETER"]
            idx_exp = param_desc.index("EXPOSURE_DURATION")
            exposure_time = float(param_values[idx_exp])
            idx_ert = param_desc.index("EXTERNAL_REPETITION_TIME")
            external_repetition_time = float(param_values[idx_ert])
        except Exception:
            pass

    meta = {
        "shape": (bands, samples, lines),
        "dtype": dtype,
        "core_null": qube.get("CORE_NULL", -32768),
        "core_low_saturation": qube.get("CORE_LOW_REPR_SATURATION", -32767),
        "core_high_saturation": qube.get("CORE_HIGH_REPR_SATURATION", -32767),
        "core_multiplier": qube.get("CORE_MULTIPLIER", 1.0),
        "core_base": qube.get("CORE_BASE", 0.0),
        "product_type": lbl_data.get("PRODUCT_TYPE", "UNKNOWN"),
        "wave_length_cen": np.array(band_bin.get("BAND_BIN_CENTER", []), dtype=np.float32),
        "wave_width": np.array(band_bin.get("BAND_BIN_WIDTH", []), dtype=np.float32),
        "wave_length_band_val": np.array(band_bin.get("BAND_BIN_ORIGINAL_BAND", []), dtype=np.int16),
        "spacecraft_solar_dist": float(qube.get("SPACECRAFT_SOLAR_DISTANCE", 441765159.0)),
        "start_time": parse_time("START_TIME"),
        "end_time": parse_time("STOP_TIME"),
        "exposure_time": exposure_time,
        "external_repetition_time": external_repetition_time,
        "mission_phase": get_str("MISSION_PHASE_NAME"),
        "solar_incidence": get_float("INCIDENCE_ANGLE"),
        "emission_angle": get_float("EMISSION_ANGLE"),
        "phase_angle": get_float("PHASE_ANGLE"),
        "sc_target_distance": get_float("TARGET_CENTER_DISTANCE"),
        "sub_spacecraft_lat": get_float("SUB_SPACECRAFT_LATITUDE"),
        "sub_spacecraft_lon": get_float("SUB_SPACECRAFT_LONGITUDE"),
        "local_hour_angle": get_float("LOCAL_HOUR_ANGLE"),
        "target_name": get_str("TARGET_NAME"),
        "target_type": get_str("TARGET_TYPE"),
    }
    qub_relative = lbl_data.get("^QUBE")
    if isinstance(qub_relative, str):
        qub_path = os.path.join(base_folder, qub_relative.strip('"'))
    else:
        raise ValueError("Could not find QUBE path in label.")
    with open(qub_path, "rb") as f:
        data = np.fromfile(f, dtype=dtype)
    data = fix_endianness(data, dtype).reshape(meta["shape"], order="F").astype(np.float32)
    null, low, high = meta["core_null"], meta["core_low_saturation"], meta["core_high_saturation"]
    flag_mask = np.zeros_like(data, dtype=np.uint8)  # 0 = valid
    flag_mask[data == null] = 1  # null
    flag_mask[data == low] = 2  # low saturation
    flag_mask[data == high] = 3  # high saturation
    flag_mask[data == -32766] = 4
    flag_mask[data == -32765] = 5
    flag_mask[data == -32764] = 6
    flag_mask[data == -32762] = 7
    flag_mask[data == -32761] = 8
    data[flag_mask != 0] = np.nan

    if not cal:
        mult, base = meta["core_multiplier"], meta["core_base"]
        if mult != 1.0 or base != 0.0:
            data *= mult
            data += base

    if return_flag_mask:
        return data, meta, flag_mask
    else:
        return data, meta

def extract_hk_data(lbl_hk_file, tab_hk_file):
    lbl_data = pvl.load(lbl_hk_file)
    columns = lbl_data["TABLE"].getlist("COLUMN")
    # Map PDS column names to internal keys
    name_map = {
        "SHUTTER STATUS": "shutter_status",
        "IR EXPO": "exposure_time_ir",
        "IR TEMP": "ir_temp",
        "CCD EXPO": "exposure_time_ccd",
        "CCD TEMP": "ccd_temp",
        "SPECT TEMP": "spect_temp",
        "TELE TEMP": "tele_temp",
        "COLD TIP TEMP": "cold_tip_temp",
        "RADIATOR TEMP": "radiator_temp",
        "LEDGE TEMP": "ledge_temp",
        "MIRROR SIN": "mirror_sin",
        "MIRROR COS": "mirror_cos",
        "START NOISY BITS": "start_noisy_bits",
        "END NOISY BITS": "end_noisy_bits",
        "NOF NOISY BITS": "num_noisy_bits",
        "CR ROW": "cr_row",
        "SUBFRAME DATA": "subframe_data",
        "SEQ STEP": "seq_step",
    }
    column_specs = {}
    for col in columns:
        name = col["NAME"].strip().upper()
        if name in name_map:
            key = name_map[name]
            start = int(col["START_BYTE"]) - 1
            length = int(col["BYTES"])
            column_specs[key] = slice(start, start + length)

    missing = set(name_map.values()) - set(column_specs.keys())
    if missing:
        print(f"Warning: Missing columns in LBL: {missing}")

    extracted_data = {key: [] for key in column_specs}

    with open(tab_hk_file) as f:
        lines = f.readlines()

    shutter_open = {"open": 1, "closed": 0}

    for line in lines:
        for key, sl in column_specs.items():
            raw = line[sl].strip()
            if key == "shutter_status":
                extracted_data[key].append(shutter_open.get(raw.lower(), None))
            else:
                try:
                    if raw.isdigit():
                        extracted_data[key].append(int(raw))
                    else:
                        extracted_data[key].append(float(raw))
                except ValueError:
                    extracted_data[key].append(None)

    shutter_data = extracted_data.get("shutter_status", [])
    closed_indexes = [i for i, val in enumerate(shutter_data) if val == 0]
    opened_indexes = [i for i, val in enumerate(shutter_data) if val == 1]

    return {
        "data": extracted_data,
        "closed_indexes": closed_indexes,
        "opened_indexes": opened_indexes,}

def load_ITF_data(filename, shape=(432, 256), dtype=np.float64):
    data = np.fromfile(filename, dtype=dtype).reshape(shape, order='F')
    return data.reshape(shape)

def parse_pds3_label(lbl_path):
    """
    Parse a PDS3 .LBL file and extract metadata for EACH TABLE object.

    Returns a list of tables, each a dict:
      {
        'pointer':   str or None     # the "^TABLE" filename, if present
        'record_bytes': int          # bytes per record (ROW_BYTES or fallback to RECORD_BYTES)
        'file_records': int          # number of rows    (ROWS     or fallback to FILE_RECORDS)
        'columns': [
            {
              'name':       str,
              'data_type':  str,
              'start_byte': int,
              'bytes':      int
            },
            ...
        ]
      }
    """
    lbl = pvl.load(lbl_path)
    pointers = lbl.get('^TABLE')
    if isinstance(pointers, str):
        pointers = [pointers]
    elif pointers is None:
        pointers = []
    tables = lbl.get('TABLE')
    if isinstance(tables, dict):
        tables = [tables]
    elif tables is None:
        raise ValueError("No TABLE object found in label")
    if len(pointers) < len(tables):
        pointers = pointers + [None] * (len(tables) - len(pointers))
    out = []
    for ptr, tbl in zip(pointers, tables):
        rec_bytes = tbl.get('ROW_BYTES', lbl.get('RECORD_BYTES'))
        row_cnt   = tbl.get('ROWS',     lbl.get('FILE_RECORDS'))
        if rec_bytes  is None or row_cnt is None:
            raise ValueError("Missing ROW_BYTES/RECORD_BYTES or ROWS/FILE_RECORDS")
        cols = tbl.get('COLUMN')
        if isinstance(cols, dict):
            cols = [cols]
        col_defs = []
        for c in cols:
            col_defs.append({
                'name':       c['NAME'],
                'data_type':  c.get('DATA_TYPE', 'ASCII_STRING'),
                'start_byte': int(c['START_BYTE']),
                'bytes':      int(c['BYTES'])
            })
        out.append({
            'pointer':      ptr,
            'record_bytes': int(rec_bytes),
            'file_records': int(row_cnt),
            'columns':      col_defs
        })
    return out
def load_pds3_tab(lbl_path, tab_path=None, table_index=0):
    """
    Load one of the tables from a PDS3 .LBL + .TAB pair.

    Parameters
    ----------
    lbl_path     : str
      Path to the .LBL file.
    tab_path     : str, optional
      If provided, uses this .TAB file; otherwise uses the '^TABLE' entry.
    table_index  : int
      Which TABLE to pick (0-based).

    Returns
    -------
    data : dict of numpy.ndarray
      Keys are column names, values are 1D arrays of length=file_records.
    """
    tables = parse_pds3_label(lbl_path)
    try:
        meta = tables[table_index]
    except IndexError:
        raise IndexError(f"Label only contains {len(tables)} table(s); index {table_index} is out of range.")
    if tab_path is None:
        if meta['pointer'] is None:
            raise ValueError("No '^TABLE' filename in label for table index "
                             f"{table_index}; please pass tab_path explicitly.")
        tab_path = meta['pointer']
    nrows = meta['file_records']
    data  = {}
    for col in meta['columns']:
        name   = col['name']
        length = col['bytes']
        dt     = col['data_type']
        if dt == 'ASCII_REAL':
            data[name] = np.empty(nrows, dtype=float)
        elif dt == 'ASCII_INTEGER':
            data[name] = np.empty(nrows, dtype=int)
        else:
            data[name] = np.empty(nrows, dtype=f'U{length}')
    with open(tab_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= nrows:
                break
            for col in meta['columns']:
                name  = col['name']
                start = col['start_byte'] - 1
                length= col['bytes']
                raw   = line[start:start+length].strip()
                arr   = data[name]
                if arr.dtype == float:
                    arr[i] = float(raw) if raw else np.nan
                elif arr.dtype == int:
                    arr[i] = int(raw)   if raw else 0
                else:
                    arr[i] = raw
    return data
#-------------------------------------------------------------------------------------
# def find_calibration_lbl(base_folder):
#     calibration_basenames = []
#     for root, _, files in os.walk(base_folder):
#         for file in files:
#             if "_HK" in file.upper():
#                 continue
#             if file.lower().endswith(".lbl"):
#                 lbl_path = os.path.join(root, file)
#                 try:
#                     metadata = parse_lbl_metadata(lbl_path)
#                     target_type = metadata.get("target_type", "").upper()
#                     if target_type == "CALIBRATION":
#                         base_name = os.path.splitext(file)[0]
#                         for ext in [".QUB", ".CUB"]:
#                             qub_path = os.path.join(root, base_name + ext)
#                             if os.path.exists(qub_path):
#                                 base_id = "_".join(base_name.split("_")[:-1])  # Remove trailing _1
#                                 calibration_basenames.append(base_id)
#                                 print(f"Found CALIBRATION: {base_id}")
#                                 break
#                         else:
#                             print(f"CALIBRATION label without matching QUB/CUB: {lbl_path}")
#                 except Exception as e:
#                     print(f"Failed to parse {lbl_path}: {e}")
#     return calibration_basenames
def find_calibration_lbl(base_folder):
    calibration_basenames = []
    for root, _, files in os.walk(base_folder):
        for file in files:
            if "_HK" in file.upper():
                continue
            if file.lower().endswith(".lbl"):
                lbl_path = os.path.join(root, file)
                try:
                    metadata = parse_lbl_metadata(lbl_path)
                    target_type = metadata.get("target_type", "").upper()
                    exposure_time = metadata.get("exposure_time", None)
                    if target_type == "CALIBRATION" and (exposure_time == 0 or exposure_time is None):
                        base_name = os.path.splitext(file)[0]
                        for ext in [".QUB", ".CUB"]:
                            qub_path = os.path.join(root, base_name + ext)
                            if os.path.exists(qub_path):
                                base_id = "_".join(base_name.split("_")[:-1])  # Remove trailing _1
                                calibration_basenames.append(base_id)
                                print(f"Found CALIBRATION with exposure_time = 0: {base_id}")
                                break
                        else:
                            print(f"CALIBRATION label without matching QUB/CUB: {lbl_path}")
                    else:
                        if exposure_time != 0 and exposure_time is not None:
                            print(f"Skipping label with exposure_time != 0: {lbl_path}")
                except Exception as e:
                    print(f"Failed to parse {lbl_path}: {e}")
    return calibration_basenames

def find_dark_qub(base_folder):
    dark_basenames = []
    for root, _, files in os.walk(base_folder):
        for file in files:
            if "_HK" in file.upper():
                continue
            if file.lower().endswith(".lbl"):
                lbl_path = os.path.join(root, file)
                try:
                    metadata = parse_lbl_metadata(lbl_path)
                    target_type = metadata.get("target_type", "").upper()
                    target_name = metadata.get("target_name", "").upper()
                    if target_type == "CALIBRATION" and target_name == "DARK":
                        base_name = os.path.splitext(file)[0]
                        for ext in [".QUB", ".CUB"]:
                            qub_path = os.path.join(root, base_name + ext)
                            if os.path.exists(qub_path):
                                base_id = "_".join(base_name.split("_")[:-1])
                                dark_basenames.append(base_id)
                                print(f"Found DARK calibration: {base_id}")
                                break
                        else:
                            print(f"DARK calibration label without matching QUB/CUB: {lbl_path}")
                except Exception as e:
                    print(f"Failed to parse {lbl_path}: {e}")
    return dark_basenames

#----------------------------------------------------------------------------------------------------
def fix_defective_pixels(cube, defective_coords, filter_band_list=None, max_iters=3):
    """
    Fix defective pixels by averaging non-defective, non-filter-boundary neighbors.

    Parameters:
    - cube: 3D numpy array, shape (bands, samples, lines)
    - defective_coords: list of tuples (sample_id, band_id) (1-based indexing!)
    - filter_band_list: optional list of integers (band_ids) that define filter boundary bands (1-based)
    - max_iters: number of iterations to fill defective pixels

    Returns:
    - corrected_cube: corrected 3D numpy array
    """
    corrected = cube.copy()
    bands, samples, lines = cube.shape
    mask = np.zeros((samples, bands), dtype=bool)  # (samples, bands)
    for sample_id, band_id in defective_coords:
        sample_idx = sample_id - 1
        band_idx = band_id - 1
        mask[sample_idx, band_idx] = True
    if filter_band_list is None:
        filter_bands_set = set()
    else:
        filter_bands_set = set(filter_band_list)
    for _ in range(max_iters):
        updated = False
        for sample_id, band_id in defective_coords:
            sample_idx = sample_id - 1
            band_idx = band_id - 1
            if not mask[sample_idx, band_idx]:
                continue
            for line_idx in range(lines):
                neighbors = []
                for ds, db in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ns, nb = sample_idx + ds, band_idx + db
                    if 0 <= ns < samples and 0 <= nb < bands:
                        if not mask[ns, nb] and (nb+1) not in filter_bands_set:
                            neighbors.append(corrected[nb, ns, line_idx])
                if neighbors:
                    corrected[band_idx, sample_idx, line_idx] = np.mean(neighbors)
                    updated = True
            mask[sample_idx, band_idx] = False  # Mark as corrected
        if not updated:
            break
    return corrected

def fit_linear_dispersion(wavelengths):
    """
    Fit a linear relation lambda = a*band + b to the given wavelength array.
    Returns (slope, intercept).
    """
    bands = np.arange(1, len(wavelengths) + 1)
    def wv(x, a, b):
        return a * x + b
    para,poc = curve_fit(wv, bands, wavelengths)
    return para,poc


def predict_dark_current(raw_cube, closed_indexes, apply_smoothing=False):
    """
    Predict the dark-current signal for each pixel (band,sample,line).

    Parameters:
        raw_cube (ndarray): Raw data cube of shape (bands, samples, lines).
        closed_indexes (list of int): Line indices where dark frames (shutter closed) occur.
        temps (ndarray): Array of length `lines` with detector temperature (K) at each line.
        exp_times (ndarray): Array of length `lines` with exposure time (sec) at each line.
        model_type (str): 'arrhenius', 'poly2', or 'dawn' (default 'arrhenius').
        apply_smoothing (bool): If True, apply Gaussian smoothing across samples on each dark frame.

    Returns:
        predicted_dark_cube (ndarray): Dark-current cube of shape (bands, samples, lines).
    """
    bands, samples, lines = raw_cube.shape
    closed = sorted(closed_indexes)
    if len(closed) == 0:
        raise ValueError("No closed (dark) frames provided.")
    dark_frames = np.array(raw_cube[..., closed], dtype=float)
    if apply_smoothing:
        for i in range(dark_frames.shape[2]):
            for b in range(bands):
                dark_frames[b, :, i] = gaussian_filter1d(dark_frames[b, :, i], sigma=1)
    if len(closed) == 1:
        dark_ref = dark_frames[:, :, 0]
        predicted = np.tile(dark_ref[:, :, np.newaxis], (1, 1, lines))
        return predicted
    predicted_dark = np.zeros((bands, samples, lines), dtype=float)
    line_idx = np.arange(lines)
    known_idx = np.array(closed)
    for b in range(bands):
        for s in range(samples):
            known_vals = dark_frames[b, s, :]
            predicted_dark[b, s, :] = np.interp(line_idx, known_idx, known_vals)
    return predicted_dark

def remove_closed_shutter_frames(raw_cube, closed_indexes):
    """
    Efficiently remove closed-shutter frames from predicted dark cube.

    Parameters:
        predicted_dark_cube (ndarray): (bands, samples, lines)
        closed_indexes (list of int): indexes of closed shutter frames

    Returns:
        dark_cube_open (ndarray): dark current for open frames only
        open_indexes (ndarray): indexes of open shutter frames
    """
    bands, samples, lines = raw_cube.shape
    mask = np.ones(lines, dtype=bool)
    mask[closed_indexes] = False
    dark_cube_open = raw_cube[:, :, mask]
    open_indexes = np.flatnonzero(mask)
    return dark_cube_open, open_indexes

def remove_indices_from_list(data_list, remove_indexes):
    """
    Remove elements from data_list at positions specified in remove_indexes.

    Parameters:
        data_list (list or ndarray): 1D list of data
        remove_indexes (list or ndarray): indexes to remove

    Returns:
        filtered_list (ndarray): data_list with elements at remove_indexes removed
    """
    data_array = np.asarray(data_list)
    mask = np.ones(data_array.shape[0], dtype=bool)
    mask[remove_indexes] = False
    filtered_list = data_array[mask]
    return filtered_list


def dn_to_radiance(dn_cube, itf, exposure_times):
    bands, samples, lines = dn_cube.shape
    assert exposure_times.shape[0] == lines
    radiance_cube = np.full_like(dn_cube, np.nan, dtype=np.float32)
    for l in range(lines):
        frame = dn_cube[:, :, l]
        expo = exposure_times[l]
        if np.isnan(expo) or expo <= 0:
            continue
        radiance_cube[:, :, l] = frame / (itf * expo)
    return radiance_cube

def radiance_to_dn(radiance_cube, itf, exposure_times):
    bands, samples, lines = radiance_cube.shape
    assert exposure_times.shape[0] == lines
    dn_cube = np.full_like(radiance_cube, np.nan, dtype=np.float32)
    for l in range(lines):
        frame = radiance_cube[:, :, l]
        expo = exposure_times[l]
        if np.isnan(expo) or expo <= 0:
            continue
        dn_cube[:, :, l] = frame * itf * expo
    return dn_cube

def reflectance(
    radiance_cube,
    solar_irradiance,
    spacecraft_solar_distance_km) -> np.ndarray:
    K = 149597870.7
    scale_factor = (np.pi * (spacecraft_solar_distance_km ** 2)) / (K ** 2)
    si = solar_irradiance[:, np.newaxis, np.newaxis]
    reflectance = (radiance_cube * scale_factor) / si
    return reflectance

def radiance_from_reflectance(
    reflectance_cube: np.ndarray,
    solar_irradiance: np.ndarray,
    spacecraft_solar_distance_km: float
) -> np.ndarray:
    K = 149597870.7  # Astronomical Unit in km
    scale_factor = (K ** 2) / (np.pi * (spacecraft_solar_distance_km ** 2))
    si = solar_irradiance[:, np.newaxis, np.newaxis]  # Shape: (bands, 1, 1)
    radiance = reflectance_cube * si * scale_factor
    return radiance

def restore_sat_nan_no(reflectance_cube, flag_mask):
    final_reflectance = reflectance_cube.copy()
    final_reflectance[flag_mask == 1] = -32768
    final_reflectance[(flag_mask == 2) | (flag_mask == 3)] = -32767
    return final_reflectance

def correct_nan_pixels_doc(refl_cube: np.ndarray, lambda_IR: np.ndarray) -> np.ndarray:
    """
    Match the exact logic from IDL Step 1 in the VIR calibration doc:
    only interpolate over NaNs *between* non-continuous segments in right_channels.
    """
    bands, samples, lines = refl_cube.shape
    corrected_cube = refl_cube.copy()
    for s in range(samples):
        for l in range(lines):
            spectrum = corrected_cube[:, s, l]
            nan_mask = ~np.isfinite(spectrum)
            valid_indices = np.where(np.isfinite(spectrum))[0]
            if len(valid_indices) < 21:
                continue  # Not enough data to fit
            for d in range(10, len(valid_indices) - 11):
                current = valid_indices[d]
                next_ = valid_indices[d + 1]

                if next_ != current + 1:
                    fit_range = valid_indices[d - 10 : d + 11]  # 21 points
                    x_fit = lambda_IR[fit_range]
                    y_fit = spectrum[fit_range]
                    nan_range = np.arange(valid_indices[d - 10], valid_indices[d + 1 + 10])
                    nan_range = nan_range[(nan_range > current) & (nan_range < next_)]
                    nan_to_fill = nan_range[nan_mask[nan_range]]
                    if len(nan_to_fill) == 0:
                        continue
                    poly = Polynomial.fit(x_fit, y_fit, deg=2).convert()
                    spectrum[nan_to_fill] = poly(lambda_IR[nan_to_fill])
            corrected_cube[:, s, l] = spectrum
    return corrected_cube


def fill_nan_with_spline(spectrum, lambda_IR, s=0.001):
    nan_mask = ~np.isfinite(spectrum)
    if np.sum(~nan_mask) == 0:
        return spectrum
    valid = np.isfinite(spectrum)
    if np.sum(valid) < 4:
        return spectrum
    spline = UnivariateSpline(lambda_IR[valid], spectrum[valid], s=1)
    spectrum[nan_mask] = spline(lambda_IR[nan_mask])
    return spectrum

def fill_cube_nan_sp(cube, lambda_IR):
    bands, samples, lines = cube.shape
    filled = cube.copy()
    for s in range(samples):
        for l in range(lines):
            filled[:, s, l] = fill_nan_with_spline(filled[:, s, l], lambda_IR)
    return filled
#--------------------------Artifact Removal-------------------#
# def destripe_wavelet_moment(image, wavelet='periodization', level=2):
#     destriped = image.copy()
#     row_means = image.mean(axis=1)
#     coeffs = pywt.wavedec(row_means, wavelet, level=level)
#     sigma = np.median(np.abs(coeffs[-1])) / 0.6745
#     uthresh = sigma * np.sqrt(2 * np.log(len(row_means)))
#     denoised_coeffs = [
#         pywt.threshold(c, value=uthresh, mode='soft') if i > 0 else c
#         for i, c in enumerate(coeffs)
#     ]
#     smoothed_means = pywt.waverec(denoised_coeffs, wavelet)[:len(row_means)]
#     for i in range(destriped.shape[0]):
#         row = destriped[i]
#         mu, sigma = row.mean(), row.std()
#         if sigma > 1e-6:
#             destriped[i] = (row - mu) / sigma * row.std() + smoothed_means[i]
#         else:
#             destriped[i] = smoothed_means[i]
#     return destriped

# def destripe_cube(cube):
#     """
#     Apply destriping to each band (sample, line) slice of a (band, sample, line) cube.
    
#     method: 'variational' or 'wavelet_moment'
#     Returns destriped cube of the same shape.
#     """
#     bands, samples, lines = cube.shape
#     destriped = np.empty_like(cube)

#     for b in range(bands):
#         slice_2d = cube[b]
#         destriped[b] = destripe_wavelet_moment(slice_2d)
#     return destriped
def destripe_wavelet_moment(image, wavelet='db4', level=2):
    """
    Destripe a 2D image using wavelet denoising of the row-wise mean.
    Ensures NO NaNs at output, even at boundaries.
    """
    destriped = image.copy()
    row_means = np.nanmean(image, axis=1)
    if np.any(np.isnan(row_means)):
        valid = ~np.isnan(row_means)
        row_means[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), row_means[valid])
    coeffs = pywt.wavedec(row_means, wavelet=wavelet, level=level, mode='periodization')
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(row_means)))
    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
    smoothed = pywt.waverec(denoised_coeffs, wavelet=wavelet, mode='periodization')
    smoothed = smoothed[:image.shape[0]]
    for i in range(image.shape[0]):
        row = destriped[i]
        mu, std = np.nanmean(row), np.nanstd(row)
        if std > 1e-6:
            destriped[i] = (row - mu) / std * std + smoothed[i]
        else:
            destriped[i] = np.full_like(row, smoothed[i])
    destriped = np.nan_to_num(destriped, nan=0.0)
    return destriped
def destripe_cube(cube):
    bands, samples, lines = cube.shape
    destriped = np.empty_like(cube)
    for b in range(bands):
        destriped[b] = destripe_wavelet_moment(cube[b])
    destriped = np.nan_to_num(destriped, nan=0.0)
    return destriped
#--------------------------Odd-even-depsike-------------------#
def get_filter_mask(nbands, filter_bands, edge_padding=0):
    """
    Create a boolean mask from a flat list of band indices.

    Parameters:
        nbands: int
        filter_bands: list of int (explicit band indices)
        edge_padding: int, expand around each index

    Returns:
        np.ndarray of shape (nbands,), dtype=bool
    """
    mask = np.zeros(nbands, dtype=bool)
    for b in filter_bands:
        for i in range(max(0, b - edge_padding), min(nbands, b + edge_padding + 1)):
            mask[i] = True
    return mask
def correct_odd_even_weighted_interp(spectrum, filter_band_mask=None):
    """
    Odd-even suppression using weighted neighbor smoothing, while skipping filter regions.
    Follows VIR_CALIBRATION_V3_1 guidelines (Section 4.4.1).
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    corrected = spectrum.copy()
    n = len(spectrum)
    if filter_band_mask is None:
        filter_band_mask = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        if not np.isfinite(spectrum[i]):
            continue
        if filter_band_mask[i]:
            prev = next = None
            for j in range(i - 1, -1, -1):
                if np.isfinite(spectrum[j]) and not filter_band_mask[j]:
                    prev = (j, spectrum[j])
                    break
            for j in range(i + 1, n):
                if np.isfinite(spectrum[j]) and not filter_band_mask[j]:
                    next = (j, spectrum[j])
                    break
            if prev and next:
                x0, y0 = prev
                x1, y1 = next
                corrected[i] = y0 + (y1 - y0) * (i - x0) / (x1 - x0)
            continue
        if np.isfinite(spectrum[i - 1]) and np.isfinite(spectrum[i + 1]):
            corrected[i] = (
                0.3 * spectrum[i - 1] + 0.4 * spectrum[i] + 0.3 * spectrum[i + 1])
    return corrected

def correct_cube_odd_even_interp(cube, filter_band_mask=None):
    """
    Apply odd-even correction across a cube using weighted interpolation.

    Parameters:
        cube (np.ndarray): Shape (bands, samples, lines)
        filter_band_mask (np.ndarray): Boolean mask for bands to skip (length = bands)

    Returns:
        np.ndarray: Odd-even corrected cube (same shape)
    """
    bands, samples, lines = cube.shape
    corrected = np.empty_like(cube)
    for s in range(samples):
        for l in range(lines):
            spectrum = cube[:, s, l]
            corrected[:, s, l] = correct_odd_even_weighted_interp(spectrum, filter_band_mask)
    return corrected
def nan_running_avg(x, window=3):
    out = np.full_like(x, np.nan, dtype=float)
    half = window // 2
    for i in range(len(x)):
        i1 = max(0, i - half)
        i2 = min(len(x), i + half + 1)
        window_vals = x[i1:i2]
        valid = np.isfinite(window_vals)
        if valid.sum() > 0:
            out[i] = np.nanmean(window_vals)
    return out
def despike_from_doc(
    spectrum,
    sigma=3.0,
    runavg_window=3,
    polyfit_window=20,
    min_valid_points=5,
    filter_band_mask=None):
    spectrum = np.asarray(spectrum, dtype=np.float64)
    cleaned = spectrum.copy()
    n_bands = len(spectrum)
    if filter_band_mask is None:
        filter_band_mask = np.zeros(n_bands, dtype=bool)
    smoothed = nan_running_avg(spectrum, window=runavg_window)
    ratio = spectrum / (smoothed + 1e-10)
    z = (ratio - np.nanmean(ratio)) / (np.nanstd(ratio) + 1e-10)
    spikes = (np.abs(z) > sigma) & (~filter_band_mask)
    spike_indices = np.where(spikes)[0]
    if len(spike_indices) == 0:
        return cleaned 
    for idx in spike_indices:
        half = polyfit_window // 2
        i1 = max(0, idx - half)
        i2 = min(n_bands, idx + half + 1)
        x_win = np.arange(i1, i2)
        y_win = spectrum[i1:i2]
        mask_win = filter_band_mask[i1:i2]
        valid = np.isfinite(y_win) & (~mask_win)
        if valid.sum() >= min_valid_points:
            try:
                coeffs = np.polyfit(x_win[valid], y_win[valid], deg=2)
                cleaned[idx] = np.polyval(coeffs, idx)
            except Exception:
                pass  # fallback below
        elif valid.sum() >= 2:
            try:
                coeffs = np.polyfit(x_win[valid], y_win[valid], deg=1)
                cleaned[idx] = np.polyval(coeffs, idx)
            except Exception:
                pass
        else:
            prev, next = None, None
            for j in range(idx - 1, -1, -1):
                if np.isfinite(spectrum[j]) and not filter_band_mask[j]:
                    prev = (j, spectrum[j])
                    break
            for j in range(idx + 1, n_bands):
                if np.isfinite(spectrum[j]) and not filter_band_mask[j]:
                    next = (j, spectrum[j])
                    break
            if prev and next:
                x0, y0 = prev
                x1, y1 = next
                cleaned[idx] = y0 + (y1 - y0) * (idx - x0) / (x1 - x0)
            else:
                cleaned[idx] = np.nan
    return cleaned


def despike_cube_from_doc(
    cube,
    sigma=3.0,
    runavg_window=3,
    polyfit_window=20,
    min_valid_points=5,
    filter_band_mask=None):
    """
    Apply despiking across a cube using official VIR method.

    Parameters:
        cube (np.ndarray): Shape (bands, samples, lines)
        filter_band_mask (np.ndarray): Boolean mask for bands to skip (length = bands)

    Returns:
        np.ndarray: Despiked cube
    """
    bands, samples, lines = cube.shape
    cleaned_cube = np.empty_like(cube)
    for s in range(samples):
        for l in range(lines):
            spectrum = cube[:, s, l]
            cleaned_cube[:, s, l] = despike_from_doc(
                spectrum,
                sigma=sigma,
                runavg_window=runavg_window,
                polyfit_window=polyfit_window,
                min_valid_points=min_valid_points,
                filter_band_mask=filter_band_mask
            )
    return cleaned_cube

#------------Visual Functiomns--------------------------------#
def show_band_image(cube, band_index, cmap_val="gray", b_to_wv_um=None, stretch=False,stretch_amount= [10, 96],
                    drop_sample_ranges=None, drop_line_ranges=None, title="", zoom_region=None, cbar_label="Reflectance",save=False):
    """
    Show a single band image from the cube with optional zoom and sample/line drop.

    Parameters:
        cube : np.ndarray
            Data cube of shape (bands, samples, lines)
        band_index : int
            Index of the band to display (1-based)
        cmap_val : str
            Colormap used for plotting
        b_to_wv_um : function
            Function that maps band index to wavelength in microns
        stretch : bool
            Whether to stretch contrast between 2nd and 98th percentile (but keep colorbar in real units)
        drop_sample_ranges : list of tuple(int, int)
            List of (start_sample, end_sample) to drop from the image (sample axis)
        drop_line_ranges : list of tuple(int, int)
            List of (start_line, end_line) to drop from the image (line axis)
        title : str
            Custom title string
        zoom_region : tuple(int, int, int, int)
            (sample_start, sample_end, line_start, line_end) to crop image region
        save : bool
            Whether to save the image to a PDF file
    """
    img = cube[band_index - 1].copy()
    if drop_sample_ranges:
        sample_mask = np.ones(img.shape[0], dtype=bool)
        for start, end in drop_sample_ranges:
            sample_mask[start:end] = False
        img = img[sample_mask, :]
    if drop_line_ranges:
        line_mask = np.ones(img.shape[1], dtype=bool)
        for start, end in drop_line_ranges:
            line_mask[start:end] = False
        img = img[:, line_mask]
    if zoom_region:
        s0, s1, l0, l1 = zoom_region
        img = img[s0:s1, l0:l1]
    if stretch:
        vmin, vmax = np.nanpercentile(img, stretch_amount)
    else:
        vmin, vmax = np.nanmin(img), np.nanmax(img)
    plt.imshow(img.T, cmap=cmap_val, origin='lower', aspect='auto',
               norm=Normalize(vmin=vmin, vmax=vmax))
    if b_to_wv_um is not None:
        band_um = b_to_wv_um(band_index)
        plt.title(f"Band {band_index} - {band_um:.4f} µm {title}")
    else:
        plt.title(f"Band {band_index} {title}")
    cbar = plt.colorbar()
    cbar.set_label(cbar_label)
    plt.xlabel("Sample")
    plt.ylabel("Line")
    if save:
        fname = title if title else f"band_{band_index}"
        plt.savefig(f"{fname}.pdf", bbox_inches='tight')
    plt.show()

def plot_superimposed_bands(
    cube,
    bands,
    colors,
    stretch=True,
    alpha=0.6,
    zoom_region=None,
    drop_sample_ranges=None,
    drop_line_ranges=None,
    title="",
    save=False):
    """
    Superimpose selected bands from a cube using specified colors.

    Parameters:
        cube : np.ndarray
            Data cube of shape (bands, samples, lines)
        bands : list of int
            Band indices to visualize (1-based)
        colors : list of str
            Matplotlib color names (e.g., "red", "green", "blue", etc.)
        stretch : bool
            Whether to apply contrast stretching (2-98 percentile)
        alpha : float
            Transparency for overlays
        zoom_region : tuple(int, int, int, int)
            (sample_start, sample_end, line_start, line_end)
        drop_sample_ranges : list of tuple(int, int)
            List of sample ranges to mask (columns)
        drop_line_ranges : list of tuple(int, int)
            List of line ranges to mask (rows)
        title : str
            Plot title
        save : bool
            Save to PDF
    """
    assert len(bands) == len(colors), "bands and colors must be same length"
    bands = [b - 1 for b in bands]
    ref_img = cube[bands[0]].copy()
    if drop_sample_ranges:
        sample_mask = np.ones(ref_img.shape[0], dtype=bool)
        for start, end in drop_sample_ranges:
            sample_mask[start:end] = False
        ref_img = ref_img[sample_mask, :]
    else:
        sample_mask = np.ones(ref_img.shape[0], dtype=bool)
    if drop_line_ranges:
        line_mask = np.ones(ref_img.shape[1], dtype=bool)
        for start, end in drop_line_ranges:
            line_mask[start:end] = False
        ref_img = ref_img[:, line_mask]
    else:
        line_mask = np.ones(ref_img.shape[1], dtype=bool)
    if zoom_region:
        s0, s1, l0, l1 = zoom_region
        ref_img = ref_img[s0:s1, l0:l1]
        sample_mask[s0:s1] = sample_mask[s0:s1]  # reduce to zoom
        line_mask[l0:l1] = line_mask[l0:l1]
    shape = ref_img.shape
    composite = np.zeros((shape[0], shape[1], 4))
    for band, color in zip(bands, colors):
        img = cube[band].copy()
        img = img[sample_mask, :]
        img = img[:, line_mask]
        if zoom_region:
            s0, s1, l0, l1 = zoom_region
            img = img[s0:s1, l0:l1]
        if stretch:
            p2, p98 = np.nanpercentile(img, [2, 98])
            img = np.clip((img - p2) / (p98 - p2 + 1e-10), 0, 1)
        else:
            min_val, max_val = np.nanmin(img), np.nanmax(img)
            img = np.clip((img - min_val) / (max_val - min_val + 1e-10), 0, 1)
        rgba = to_rgba(color, alpha=alpha)
        for c in range(3):
            composite[..., c] += img * rgba[c]
        composite[..., 3] = 1.0
    composite[..., :3] = np.clip(composite[..., :3], 0, 1)
    plt.figure(figsize=(8, 6))
    plt.imshow(composite.transpose(1, 0, 2), origin="lower", aspect="auto")
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Line")
    if save:
        fname = title if title else "superimposed_bands"
        plt.savefig(f"{fname}.pdf", bbox_inches="tight")
    plt.show()
#----------------------------------------------------------------------------------------------------------
# def fit_dark_thermal_model(cube, exposure_times, ir_temperatures, closed_indices, wavelengths, itf):
#     """
#     Fit a per-pixel dark current + thermal background model for a VIR IR cube.

#     The model assumes that the dark signal at each (band, sample) pixel is given by:
#         DN = exp(a / T + b) + c * B(λ, T) * t_int * ITF
#     where:
#         - T is the IR sensor temperature (in Kelvin)
#         - t_int is the integration time (in seconds)
#         - B(λ, T) is the Planck radiance at wavelength λ and temperature T
#         - ITF is the instrument transfer function at that pixel

#     Parameters:
#     ----------
#     cube : np.ndarray, shape (bands, samples, lines)
#         Calibrated dark cube data in DN.
#     exposure_times : np.ndarray, shape (lines,)
#         Integration times per line (in seconds).
#     ir_temperatures : np.ndarray, shape (lines,)
#         Spectrometer temperatures per line (in Kelvin).
#     closed_indices : list or array of int
#         Indices of lines that correspond to closed-shutter dark frames.
#     wavelengths : np.ndarray, shape (bands,)
#         Center wavelengths for each band (in microns).
#     itf : np.ndarray, shape (bands, samples)
#         Instrument transfer function values (DN/e⁻).

#     Returns:
#     -------
#     a_map : np.ndarray, shape (bands, samples)
#         Fitted coefficient "a" (related to 1/T dependence of dark current).
#     b_map : np.ndarray, shape (bands, samples)
#         Fitted coefficient "b" (logarithmic intercept).
#     c_map : np.ndarray, shape (bands, samples)
#         Fitted coefficient "c" (scales thermal background component).
#     """
#     bands, samples, lines = cube.shape
#     closed_indices = np.asarray(closed_indices)
#     temperature_arr = np.asarray(ir_temperatures)[closed_indices]
#     integration_time_arr = np.asarray(exposure_times)[closed_indices]
#     num_closed = len(closed_indices)
#     a_map = np.full((bands, samples), np.nan)
#     b_map = np.full((bands, samples), np.nan)
#     c_map = np.full((bands, samples), np.nan)
#     def planck_radiance(wavelength_um, temperature):
#         """Compute Planck radiance for given wavelength (microns) and temperature (K)."""
#         wavelength_m = wavelength_um * 1e-6
#         numerator = 2 * planck_constant * speed_of_light**2 / wavelength_m**5
#         exponent = planck_constant * speed_of_light / (wavelength_m * boltzmann_constant * temperature)
#         return numerator / (np.exp(exponent) - 1)
#     def model_residuals(params, temperature, thermal_term, observed_dn):
#         """Residuals between model and observed dark DN values."""
#         a_coef, b_coef, c_coef = params
#         model_dn = np.exp(a_coef / temperature + b_coef) + c_coef * thermal_term
#         return model_dn - observed_dn
#     successful_fits = 0
#     for band_index in range(bands):
#         wavelength = wavelengths[band_index]
#         planck_vals = planck_radiance(wavelength, temperature_arr)
#         for sample_index in range(samples):
#             itf_value = itf[band_index, sample_index]
#             thermal_component = planck_vals * integration_time_arr * itf_value
#             observed_dn = cube[band_index, sample_index, closed_indices]
#             if not np.all(np.isfinite(observed_dn)) or np.any(thermal_component <= 0):
#                 continue
#             try:
#                 result = least_squares(
#                     model_residuals,
#                     x0=[-1000, 0, 1],
#                     bounds=([-1e4, -100, 0], [0, 100, 1e4]),
#                     args=(temperature_arr, thermal_component, observed_dn),
#                     loss='soft_l1',
#                     max_nfev=5000
#                 )
#                 a_fit, b_fit, c_fit = result.x

#                 if result.success and a_fit < 0 and c_fit >= 0:
#                     a_map[band_index, sample_index] = a_fit
#                     b_map[band_index, sample_index] = b_fit
#                     c_map[band_index, sample_index] = c_fit
#                     successful_fits += 1
#             except Exception:
#                 continue
#     print(f"Total successful fits: {successful_fits} / {bands * samples}")
#     return a_map, b_map, c_map

def fit_dark_thermal_model(cube, exposure_times, ir_temperatures, spec_temperatures,
                           closed_indices, wavelengths, itf):
    """
    Fit a per-pixel dark current + thermal background model for a VIR IR cube.

    Parameters
    ----------
    cube : np.ndarray, shape (bands, samples, lines)
        Cube with dark + bias frames. First 5 lines are bias frames.
    exposure_times : np.ndarray, shape (lines,)
        Integration times for each line.
    ir_temperatures : np.ndarray, shape (lines,)
        Detector IR temperature for each line.
    spec_temperatures : np.ndarray, shape (lines,)
        Spectrometer (optics) temperature for Planck term.
    closed_indices : list[int]
        Line indices of the dark frames.
    wavelengths : np.ndarray, shape (bands,)
        Wavelength centers in microns.
    itf : np.ndarray, shape (bands, samples)
        Instrument transfer function.

    Returns
    -------
    a_map, b_map, c_map : np.ndarray, shape (bands, samples)
        Fitted parameter maps for exp and thermal model.
    """
    bands, samples, lines = cube.shape
    closed_indices = np.asarray(closed_indices)
    n_bias = 5
    n_dark = len(closed_indices)

    T_dark = np.asarray(ir_temperatures)[closed_indices]
    T_spec = np.asarray(spec_temperatures)[closed_indices]
    t_int = np.asarray(exposure_times)[closed_indices]

    a_map = np.full((bands, samples), np.nan)
    b_map = np.full((bands, samples), np.nan)
    c_map = np.full((bands, samples), np.nan)

    def planck_radiance(wavelength_um, temperature):
        wavelength_m = wavelength_um * 1e-6
        numerator = 2 * planck_constant * speed_of_light**2 / wavelength_m**5
        exponent = planck_constant * speed_of_light / (wavelength_m * boltzmann_constant * temperature)
        return numerator / (np.exp(exponent) - 1)

    def model_residuals(params, T_dark, thermal_term, observed_dn):
        a, b, c = params
        model = np.exp(a / T_dark + b) + c * thermal_term
        return model - observed_dn

    bias_cube = cube[:, :, :n_bias]

    successful_fits = 0

    for b in range(bands):
        λ = wavelengths[b]
        try:
            planck_vals = planck_radiance(λ, T_spec)
        except FloatingPointError:
            continue

        for s in range(samples):
            itf_val = itf[b, s]
            thermal_term = planck_vals * t_int * itf_val

            dark_vals = np.zeros(n_dark)
            for i in range(n_dark):
                bias_line = i % n_bias
                dark_vals[i] = cube[b, s, closed_indices[i]] - bias_cube[b, s, bias_line]

            if not np.all(np.isfinite(dark_vals)) or np.any(thermal_term <= 0):
                continue

            try:
                result = least_squares(
                    model_residuals,
                    x0=[-1000, 0, 1],
                    bounds=([-1e4, -100, 0], [0, 100, 1e4]),
                    args=(T_dark, thermal_term, dark_vals),
                    loss='soft_l1',
                    max_nfev=5000
                )
                a_fit, b_fit, c_fit = result.x
                if result.success and a_fit < 0 and c_fit >= 0:
                    a_map[b, s] = a_fit
                    b_map[b, s] = b_fit
                    c_map[b, s] = c_fit
                    successful_fits += 1
            except Exception:
                continue
    print(f"Total successful fits: {successful_fits} / {bands * samples}")
    return a_map, b_map, c_map

def predict_dark_cube_from_model(a_map, b_map, c_map, exposure_times, ir_temperatures, spectral_temperatures, wavelengths, itf):
    """
    Predict dark current cube from model parameters (a, b, c), using separate temperatures
    for the dark current and thermal background components.

    Parameters:
        a_map, b_map, c_map      : (bands, samples) arrays from model fit
        exposure_times           : (lines,) array of integration times (seconds)
        ir_temperatures          : (lines,) array of IR detector temperatures (K) [used in exp(a/T + b)]
        spectral_temperatures    : (lines,) array of spectrometer temperatures (K) [used in Planck]
        wavelengths              : (bands,) center wavelengths (microns)
        itf                      : (bands, samples) Instrument Transfer Function (DN/e⁻)

    Returns:
        dark_cube_predicted : (bands, samples, lines) array with predicted dark current values
    """
    # Physical constants
    h = 6.62607015e-34  # Planck constant (J·s)
    c = 2.99792458e8    # Speed of light (m/s)
    k = 1.380649e-23    # Boltzmann constant (J/K)

    bands, samples = a_map.shape
    lines = len(exposure_times)
    dark_cube = np.full((bands, samples, lines), np.nan)

    T_dark = np.asarray(ir_temperatures)         # For exp(a/T + b)
    T_spec = np.asarray(spectral_temperatures)   # For Planck
    t_line = np.asarray(exposure_times)

    for b in range(bands):
        λ = wavelengths[b] * 1e-6  # microns → meters
        factor1 = 2 * h * c**2 / λ**5
        exponent = h * c / (λ * k * T_spec)
        B_lambda_Tspec = factor1 / (np.exp(exponent) - 1)  # shape (lines,)

        for s in range(samples):
            a = a_map[b, s]
            b_ = b_map[b, s]
            c_ = c_map[b, s]
            itf_val = itf[b, s]

            if not np.isfinite(a) or not np.isfinite(c_):
                continue
            exp_term = np.exp(a / T_dark + b_)                  # shape (lines,)
            thermal_term = c_ * B_lambda_Tspec * t_line * itf_val  # shape (lines,)
            dark_cube[b, s, :] = exp_term + thermal_term
    return dark_cube

def bias_correct_dark_cube(dark_cube_pred, cube_actual, closed_indices):
    """
    Apply median bias correction to predicted dark cube using residuals from known dark lines.

    Parameters:
        dark_cube_pred : np.ndarray of shape (bands, samples, lines)
            The predicted dark current cube from the model.
        cube_actual : np.ndarray of shape (bands, samples, lines)
            The actual data cube (with dark frames in closed_indices).
        closed_indices : list or np.ndarray
            Line indices where closed-shutter (dark) frames are available.

    Returns:
        dark_cube_corrected : np.ndarray of shape (bands, samples, lines)
            Bias-corrected dark current prediction.
        bias_correction : np.ndarray of shape (bands, samples)
            The median residual (actual - predicted) used for correction.
    """
    dark_actual = cube_actual[:, :, closed_indices]
    dark_model_subset = dark_cube_pred[:, :, closed_indices]
    residual = dark_actual - dark_model_subset
    bias_correction = np.nanmedian(residual, axis=2)
    dark_cube_corrected = dark_cube_pred + bias_correction[:, :, np.newaxis]
    return dark_cube_corrected, bias_correction



def sharpen_science_cube_with_star_psf(science_cube, star_cube, band_idx=None,
                                       star_position=None, patch_size=15, iterations=20):
    """
    Automatically estimate PSF from a star_cube and use it to sharpen a science_cube.
    
    Parameters:
    - science_cube: numpy array of shape (band, sample, line)
    - star_cube:    numpy array of same shape (band, sample, line)
    - band_idx:     band index for PSF extraction (if None, uses band with max signal)
    - star_position: (sample, line) of the star center. If None, auto-detects brightest.
    - patch_size:   size of square patch to crop around the star
    - iterations:   number of iterations for Richardson-Lucy
    
    Returns:
    - sharpened_cube: numpy array of shape (band, sample, line)
    - star_patch:     2D numpy array of shape (patch_size, patch_size) cropped around the star
    - band_idx:       The band index used for PSF estimation
    """
    
    def two_d_gaussian(xy, x0, y0, sigma_x, sigma_y, amplitude, offset):
        x, y = xy
        g = amplitude * np.exp(-(((x - x0)**2) / (2 * sigma_x**2) +
                                 ((y - y0)**2) / (2 * sigma_y**2))) + offset
        return g.ravel()
    def fit_psf(star_patch):
        star_patch = np.nan_to_num(star_patch)
        y, x = np.indices(star_patch.shape)
        max_idx = np.unravel_index(np.argmax(star_patch), star_patch.shape)
        x0_guess, y0_guess = max_idx[1], max_idx[0]
        
        initial_guess = (
            x0_guess, y0_guess,
            2, 2,
            np.max(star_patch),
            np.min(star_patch))
        bounds = (
            [0, 0, 0.5, 0.5, 0, -np.inf],
            [star_patch.shape[1], star_patch.shape[0], 5, 5, np.inf, np.inf])
        try:
            popt, _ = curve_fit(two_d_gaussian, (x, y), star_patch.ravel(),
                                p0=initial_guess, bounds=bounds, maxfev=5000)
        except RuntimeError:
            raise RuntimeError("PSF fit failed. Try a cleaner star patch or better guess.")
        return popt
    def create_psf_kernel(size, sigma_x, sigma_y):
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-((xx**2) / (2. * sigma_x**2) + (yy**2) / (2. * sigma_y**2)))
        return kernel / np.sum(kernel)
    if band_idx is None:
        summed = np.sum(star_cube, axis=(1,2))  # Sum over spatial axes
        band_idx = np.argmax(summed)
    star_image = star_cube[band_idx]  # shape: (sample, line)
    if star_position is None:
        star_position = np.unravel_index(np.nanargmax(star_image), star_image.shape)
    sample_c, line_c = star_position
    half = patch_size // 2
    star_patch = star_image[
        max(sample_c - half, 0):sample_c + half + 1,
        max(line_c - half, 0):line_c + half + 1]
    if star_patch.shape[0] < patch_size or star_patch.shape[1] < patch_size:
        raise ValueError("Patch size too large or star too close to the edge.")
    x0, y0, sigma_x, sigma_y, amp, offset = fit_psf(star_patch)
    psf_kernel = create_psf_kernel(size=patch_size, sigma_x=sigma_x, sigma_y=sigma_y)
    def fill_nan(image, fill_value=0.0):
        return np.nan_to_num(image, nan=fill_value, posinf=fill_value, neginf=fill_value)
    sharpened_cube = np.empty_like(science_cube)
    for b in range(science_cube.shape[0]):
        image = fill_nan(science_cube[b])
        sharpened_cube[b] = richardson_lucy(image, psf_kernel, num_iter=iterations)
    return sharpened_cube, star_patch, band_idx
